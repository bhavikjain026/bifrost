package main

// Best-Before: time-sensitive cached answers get a review deadline. After it, a cache
// hit is checked claim by claim against live web sources (the freshness checker
// service, which uses Tavily) BEFORE it is served. Each checked claim ends up
// supported, contradicted or unconfirmed:
//   - contradicted (sources state a different value): the cache entry is deleted and
//     the answer is regenerated with the source lines added (by the small companion
//     plugin "cacheguard-evidence"), and the new answer is cached in its place;
//   - unconfirmed (a fast-changing claim the sources neither confirm nor contradict):
//     this is not evidence the answer is wrong, so it is never called out of date. If
//     a search on the question found relevant lines, the answer is regenerated from
//     them, and the model is told to say so if they don't state the answer
//     (on_unconfirmed "regenerate", the default). Otherwise it is served, marked
//     unconfirmed, and checked again sooner;
//   - supported: served, and the review deadline restarts.
//
// New time-sensitive answers are also checked in the background right after they are
// cached (check_on_store), so a stale answer does not stay in the cache.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/plugins/cacheguard/bestbefore"
)

// BestBeforeConfig is the "best_before" block of the plugin config.
type BestBeforeConfig struct {
	Enabled      bool           `json:"enabled"`
	CheckerURL   string         `json:"checker_url"`    // default http://127.0.0.1:8098/check
	TimeoutMs    int            `json:"timeout_ms"`     // whole freshness check, default 30000
	TTLSeconds   map[string]int `json:"ttl_seconds"`    // per-category review deadline overrides
	CheckOnStore *bool          `json:"check_on_store"` // check new time-sensitive answers in the background (default true)
	OnCheckError string         `json:"on_check_error"` // "regenerate" (default) or "serve"
	// What to do when a fast-changing claim can be neither confirmed nor contradicted:
	// "regenerate" (default) from search results for the question, if any were found;
	// "serve" the cached answer, marked unconfirmed.
	OnUnconfirmed string `json:"on_unconfirmed"`

	// Refresh-ahead: a hit in the last part of an answer's review window starts a
	// background check (and, if stale, a background regeneration), so users rarely
	// reach an expired answer and wait for a live check.
	RefreshAhead  *bool   `json:"refresh_ahead"`  // default true
	RefreshWindow float64 `json:"refresh_window"` // fraction of the review window, default 0.2

	// Stale-rate target (slo.go): one number instead of per-category deadlines. The
	// review interval per category comes from how often such answers change, and
	// answers the detector does not flag are reviewed too. 0 = off (fixed deadlines).
	MaxStaleRate    float64 `json:"max_stale_rate"`    // e.g. 0.02 = at most ~2% of hits stale
	CheckerRecall   float64 `json:"checker_recall"`    // share of real changes the checker catches, default 1
	ChangeRatesFile string  `json:"change_rates_file"` // optional: rates learned from your own traffic
}

// Shared with the cacheguard-evidence plugin through the request context.
// Only string keys and standard-library types cross between the two plugins.
const (
	evidenceTextKey = schemas.BifrostContextKey("cacheguard-evidence-text") // string: evidence to add for this request
	evidenceDoneKey = schemas.BifrostContextKey("cacheguard-evidence-done") // *atomic.Bool: set true once added
	evidenceHeader  = "x-cacheguard-evidence"                               // base64 evidence on the regeneration request
)

var (
	adminAuth    = adminAuthorization()             // for Bifrost's cache-delete API when dashboard login is on
	setupToken   = os.Getenv("BIFROST_SETUP_TOKEN") // OSS Bifrost locks /api with this token until dashboard auth is on
	storeChecks  = make(chan struct{}, 2)           // at most 2 background checks at once
	checkerHTTP  = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	timeNow      = time.Now
	checkerQueue atomic.Int64 // background checks waiting or running (visible in tests)
)

// adminAuthorization: CACHEGUARD_ADMIN_AUTH (a full Authorization value), or
// CACHEGUARD_ADMIN_USER + CACHEGUARD_ADMIN_PASSWORD (sent as HTTP Basic, which Bifrost
// accepts for its admin API when dashboard login is enabled).
func adminAuthorization() string {
	if v := os.Getenv("CACHEGUARD_ADMIN_AUTH"); v != "" {
		return v
	}
	u, p := os.Getenv("CACHEGUARD_ADMIN_USER"), os.Getenv("CACHEGUARD_ADMIN_PASSWORD")
	if u == "" || p == "" {
		return ""
	}
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(u+":"+p))
}

func initBestBefore() error {
	bb := &cfg.BestBefore
	if bb.CheckerURL == "" {
		bb.CheckerURL = "http://127.0.0.1:8098/check"
	}
	if bb.TimeoutMs <= 0 {
		bb.TimeoutMs = 30_000
	}
	if bb.OnCheckError == "" {
		bb.OnCheckError = "regenerate"
	}
	if bb.OnCheckError != "regenerate" && bb.OnCheckError != "serve" {
		return fmt.Errorf("cacheguard: best_before.on_check_error must be \"regenerate\" or \"serve\", got %q", bb.OnCheckError)
	}
	if bb.OnUnconfirmed == "" {
		bb.OnUnconfirmed = "regenerate"
	}
	if bb.OnUnconfirmed != "regenerate" && bb.OnUnconfirmed != "serve" {
		return fmt.Errorf("cacheguard: best_before.on_unconfirmed must be \"regenerate\" or \"serve\", got %q", bb.OnUnconfirmed)
	}
	if bb.CheckOnStore == nil {
		t := true
		bb.CheckOnStore = &t
	}
	if bb.RefreshAhead == nil {
		t := true
		bb.RefreshAhead = &t
	}
	if bb.RefreshWindow <= 0 || bb.RefreshWindow >= 1 {
		bb.RefreshWindow = 0.2
	}
	if err := initSLO(); err != nil {
		return err
	}
	u, err := url.Parse(bb.CheckerURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("cacheguard: best_before.checker_url must be an http(s) URL, got %q", bb.CheckerURL)
	}
	if h := u.Hostname(); !cfg.AllowRemoteLoopback && h != "127.0.0.1" && h != "localhost" && h != "::1" {
		return fmt.Errorf("cacheguard: best_before.checker_url must be on this machine (localhost); set allow_remote_loopback to override")
	}
	return nil
}

// evidenceLine is one line quoted from a web source.
type evidenceLine struct {
	Line      string `json:"line"`
	URL       string `json:"url"`
	Published string `json:"published"`
}

// checkResult mirrors the freshness checker's POST /check response (fields we use).
// Label is "stale" only for a contradicted claim; Urgent marks fast-changing facts.
type checkResult struct {
	Claims []struct {
		Text       string         `json:"text"`
		Flagged    bool           `json:"flagged"`
		Verdict    string         `json:"verdict"` // supported | contradicted | not-enough-info | not-checked
		Label      string         `json:"label"`
		Score      int            `json:"score"`
		Urgent     bool           `json:"urgent"`
		NewerValue string         `json:"newer_value"`
		Evidence   []evidenceLine `json:"evidence"`
	} `json:"claims"`
	// Lines from one search on the question itself, sent when a fast-changing claim
	// was unconfirmed and nothing was contradicted.
	QuestionEvidence []evidenceLine `json:"question_evidence"`
	Summary          struct {
		Flagged     int  `json:"flagged"`
		Checked     int  `json:"checked"`
		Stale       int  `json:"stale"`       // contradicted claims
		Unconfirmed int  `json:"unconfirmed"` // fast-changing claims neither confirmed nor contradicted
		MinScore    *int `json:"min_score"`
	} `json:"summary"`
}

// regenPlan says whether a checked answer must be regenerated, and with which
// evidence. reason is "" when the answer may be served.
func (r *checkResult) regenPlan() (reason, detail, evidence string) {
	switch {
	case r.Summary.Stale > 0:
		detail, evidence = r.staleDetail()
		return "stale-claim", detail, evidence
	case r.Summary.Unconfirmed > 0 && cfg.BestBefore.OnUnconfirmed == "regenerate" && len(r.QuestionEvidence) > 0:
		detail, evidence = r.unconfirmedDetail()
		return "unconfirmed-claim", detail, evidence
	}
	return "", "", ""
}

func shortDate(s string) string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

// unconfirmedDetail: evidence for regenerating an answer whose fast-changing claim
// could not be confirmed. It gives the search results for the question and never says
// the earlier answer was wrong.
func (r *checkResult) unconfirmedDetail() (detail, evidence string) {
	today := timeNow().UTC().Format("2006-01-02")
	var details []string
	for _, c := range r.Claims {
		if c.Verdict == "not-enough-info" && c.Urgent {
			details = append(details, "unconfirmed: "+c.Text)
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Web search results for this question (retrieved %s). An earlier answer could not be confirmed or ruled out from sources; that alone does not mean it was wrong.\n", today)
	for i, e := range r.QuestionEvidence {
		if i == 6 || b.Len() > 1500 {
			break
		}
		fmt.Fprintf(&b, "- Source (%s, %s): %q\n", e.URL, shortDate(e.Published), e.Line)
	}
	fmt.Fprintf(&b, "Use these results only where they state the answer directly, and mention the source date. If they do not state it, give your best answer and say that you could not confirm it is current as of %s.\n", today)
	ev := b.String()
	if len(ev) > 2000 {
		ev = ev[:2000]
	}
	return strings.Join(details, "; "), ev
}

// unconfirmedRetry: how soon to check again an answer that could not be confirmed
// and has no better sources yet (a quarter of its review interval, 10 min to 6 h).
func unconfirmedRetry(ttl time.Duration) time.Duration {
	r := ttl / 4
	if r < 10*time.Minute {
		r = 10 * time.Minute
	}
	if r > 6*time.Hour {
		r = 6 * time.Hour
	}
	if ttl > 0 && r > ttl {
		r = ttl
	}
	return r
}

func checkFreshness(answer, question string) (*checkResult, error) {
	payload, _ := json.Marshal(map[string]string{"answer": answer, "question": question})
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.BestBefore.TimeoutMs)*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BestBefore.CheckerURL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := checkerHTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("checker status %d", res.StatusCode)
	}
	var out checkResult
	if err := json.NewDecoder(io.LimitReader(res.Body, 4<<20)).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

// staleDetail is a short header-safe summary; evidence is the text given to the model
// when regenerating (only lines that came from the sources, never model text). Only
// contradicted claims (label "stale") are listed.
func (r *checkResult) staleDetail() (detail, evidence string) {
	var details []string
	var b strings.Builder
	b.WriteString("Up-to-date facts from live web sources (checked ")
	b.WriteString(timeNow().UTC().Format("2006-01-02"))
	b.WriteString("). Base your answer on these facts; do not repeat older values.\n")
	for _, c := range r.Claims {
		if c.Label != "stale" {
			continue
		}
		if c.NewerValue != "" {
			details = append(details, fmt.Sprintf("%s -> %s", c.Text, c.NewerValue))
		} else {
			details = append(details, c.Text)
		}
		if c.NewerValue != "" {
			fmt.Fprintf(&b, "- Newest value found in the sources: %s (an older answer said: %q).\n", c.NewerValue, c.Text)
		} else {
			fmt.Fprintf(&b, "- Sources describe this older statement as no longer current: %q.\n", c.Text)
		}
		for i, e := range c.Evidence {
			if i == 2 {
				break
			}
			fmt.Fprintf(&b, "  Source (%s, %s): %q\n", e.URL, shortDate(e.Published), e.Line)
		}
		if b.Len() > 1800 {
			break
		}
	}
	ev := b.String()
	if len(ev) > 2000 {
		ev = ev[:2000]
	}
	return strings.Join(details, "; "), ev
}

// answerText returns the text of a chat or text-completion response.
func answerText(resp *schemas.BifrostResponse) string {
	switch {
	case resp.ChatResponse != nil:
		for _, ch := range resp.ChatResponse.Choices {
			if ch.ChatNonStreamResponseChoice != nil && ch.ChatNonStreamResponseChoice.Message != nil {
				if t, ok := messageText(*ch.ChatNonStreamResponseChoice.Message); ok {
					return t
				}
			}
		}
	case resp.TextCompletionResponse != nil:
		for _, ch := range resp.TextCompletionResponse.Choices {
			if ch.TextCompletionResponseChoice != nil && ch.TextCompletionResponseChoice.Text != nil {
				return *ch.TextCompletionResponseChoice.Text
			}
		}
	}
	return ""
}

// reviewDeadline: when a cached answer to this question must be re-checked.
// ok=false means it is never reviewed: not time-sensitive by our rules (and no
// max_stale_rate), or outside the detector's scope.
func reviewDeadline(prompt string, cached storedPrompt, known bool) (time.Time, time.Duration, bool) {
	r := bestbefore.Detect(prompt, timeNow())
	var ttl time.Duration
	switch {
	case r.Status == bestbefore.Flagged:
		ttl = bestbefore.TTLFor(r, cfg.BestBefore.TTLSeconds)
		if iv, ok := sloInterval(r.Category); ok {
			ttl = iv
		}
	case r.Status == bestbefore.NotFlagged:
		iv, ok := sloInterval(notFlaggedGroup) // only with max_stale_rate on
		if !ok {
			return time.Time{}, 0, false
		}
		ttl = iv
	default:
		return time.Time{}, 0, false
	}
	if !known {
		return time.Time{}, ttl, true // age unknown (e.g. after a restart): check now
	}
	return lastReview(cached).Add(ttl), ttl, true
}

// reviewCategory: the detector category, or the not-flagged group ("" if out of scope).
func reviewCategory(prompt string) string {
	switch r := bestbefore.Detect(prompt, timeNow()); r.Status {
	case bestbefore.Flagged:
		return r.Category
	case bestbefore.NotFlagged:
		return notFlaggedGroup
	}
	return ""
}

// lastReview: when the cached answer was last known fresh (stored or re-checked).
func lastReview(p storedPrompt) time.Time {
	if p.checkedAt.After(p.cachedAt) {
		return p.checkedAt
	}
	return p.cachedAt
}

// bestBeforeOnHit runs after the compatibility checks passed. It may turn the
// decision into a reject ("stale-claim") with evidence for the regeneration.
func bestBeforeOnHit(d *decision, resp *schemas.BifrostResponse, prompt string, cached storedPrompt, known bool, raw *rawRequest) {
	if !cfg.BestBefore.Enabled || prompt == "" {
		return
	}
	deadline, ttl, sensitive := reviewDeadline(prompt, cached, known)
	if !sensitive {
		d.Freshness = "not-time-sensitive"
		return
	}
	d.FreshDeadline = deadline
	d.ReviewEvery, d.ReviewBasis = ttl, "ttl"
	if sloIntervals != nil && ttl != 0 {
		if cat := reviewCategory(prompt); cat != "" {
			if iv, ok := sloInterval(cat); ok && iv == ttl {
				d.ReviewBasis = "max-stale-rate"
			}
		}
	}
	switch {
	case known && cached.stale:
		d.Action, d.Reason, d.Freshness = "reject", "known-stale", "stale"
		if cached.unconfirmed {
			d.Reason, d.Freshness = "known-unconfirmed", "unconfirmed"
		}
		d.Evidence = cached.evidence
		return
	case !deadline.IsZero() && timeNow().Before(deadline):
		d.Freshness = "within-deadline"
		if *cfg.BestBefore.RefreshAhead && deadline.Sub(timeNow()) < time.Duration(float64(ttl)*cfg.BestBefore.RefreshWindow) {
			if startRefresh(d.CacheID, prompt, answerText(resp), raw, lastReview(cached)) {
				d.Refresh = "started"
			}
		}
		return
	}
	answer := answerText(resp)
	if answer == "" {
		d.Freshness = "check-failed"
		if cfg.BestBefore.OnCheckError == "regenerate" {
			d.Action, d.Reason = "reject", "freshness-check-failed"
		}
		return
	}
	t0 := timeNow()
	res, err := checkFreshness(answer, prompt)
	d.CheckLatency = timeNow().Sub(t0)
	if err != nil {
		d.Freshness, d.Detail = "check-failed", err.Error()
		if cfg.BestBefore.OnCheckError == "regenerate" {
			d.Action, d.Reason = "reject", "freshness-check-failed"
		}
		return
	}
	d.ClaimsChecked, d.StaleClaims, d.UnconfirmedClaims = res.Summary.Checked, res.Summary.Stale, res.Summary.Unconfirmed
	if res.Summary.MinScore != nil {
		d.MinScore = *res.Summary.MinScore
	}
	if reason, detail, ev := res.regenPlan(); reason != "" {
		d.Action, d.Reason, d.Detail, d.Evidence = "reject", reason, detail, ev
		d.Freshness = "stale"
		if reason == "unconfirmed-claim" {
			d.Freshness = "unconfirmed"
		}
		unconfirmed := reason == "unconfirmed-claim"
		store.update(d.CacheID, func(p *storedPrompt) { p.stale, p.evidence, p.unconfirmed = true, ev, unconfirmed })
		return
	}
	if res.Summary.Unconfirmed > 0 {
		// Could not confirm, and nothing better was found: serve it, marked
		// unconfirmed, and check again sooner instead of restarting the full deadline.
		retry := unconfirmedRetry(ttl)
		d.Freshness = "unconfirmed"
		d.FreshDeadline = timeNow().Add(retry)
		store.update(d.CacheID, func(p *storedPrompt) { p.checkedAt = timeNow().Add(retry - ttl) })
		return
	}
	d.Freshness = "rechecked-fresh"
	d.FreshDeadline = time.Time{}
	if deadline2, _, ok := reviewDeadline(prompt, storedPrompt{cachedAt: timeNow()}, true); ok {
		d.FreshDeadline = deadline2
	}
	store.update(d.CacheID, func(p *storedPrompt) { p.checkedAt = timeNow() })
}

var refreshing sync.Map // cache id -> struct{}: one background refresh per entry at a time

// startRefresh checks a cached answer in the background before its deadline. If a
// claim is stale, the entry is marked (so no hit serves it), removed, and the
// request is regenerated in the background with the fresh evidence, so the next
// user gets the corrected answer from the cache. Returns false if a refresh for
// this entry is already running, or one finished after `seen` (the last review this
// request saw), so a request holding an older copy of the entry doesn't refresh twice.
func startRefresh(cacheID, prompt, answer string, raw *rawRequest, seen time.Time) bool {
	if cacheID == "" || answer == "" {
		return false
	}
	if _, busy := refreshing.LoadOrStore(cacheID, struct{}{}); busy {
		return false
	}
	if p, ok := store.get(cacheID); ok && lastReview(p).After(seen) {
		refreshing.Delete(cacheID)
		return false
	}
	checkerQueue.Add(1)
	go func() {
		defer checkerQueue.Add(-1)
		defer refreshing.Delete(cacheID)
		storeChecks <- struct{}{}
		defer func() { <-storeChecks }()
		res, err := checkFreshness(answer, prompt)
		if err != nil {
			dbg("refresh-ahead %s: check failed (%v); the deadline check will retry", cacheID, err)
			return
		}
		reason, _, evidence := res.regenPlan()
		if reason == "" {
			if res.Summary.Unconfirmed > 0 {
				dbg("refresh-ahead %s: unconfirmed and no better sources; the deadline check will retry", cacheID)
				return
			}
			store.update(cacheID, func(p *storedPrompt) { p.checkedAt = timeNow() })
			dbg("refresh-ahead %s: still fresh, deadline extended", cacheID)
			return
		}
		unconfirmed := reason == "unconfirmed-claim"
		store.update(cacheID, func(p *storedPrompt) { p.stale, p.evidence, p.unconfirmed = true, evidence, unconfirmed })
		if raw == nil || cfg.Mode != "enforce" {
			return
		}
		keep := deleteCacheEntry(cacheID) == nil
		status, _, _, err := loopback(nil, raw, evidence, keep)
		dbg("refresh-ahead %s: %s, removed=%v, regenerated status=%d err=%v", cacheID, reason, keep, status, err)
	}()
	return true
}

// checkOnStore checks a newly cached time-sensitive answer in the background. An answer
// with a contradicted claim is marked (so any hit regenerates) and deleted from the
// cache. An unconfirmed claim does not delete a freshly generated answer.
func checkOnStore(cacheID, prompt, answer string) {
	if !cfg.BestBefore.Enabled || !*cfg.BestBefore.CheckOnStore || answer == "" {
		return
	}
	if r := bestbefore.Detect(prompt, timeNow()); r.Status != bestbefore.Flagged {
		return
	}
	checkerQueue.Add(1)
	go func() {
		defer checkerQueue.Add(-1)
		storeChecks <- struct{}{}
		defer func() { <-storeChecks }()
		res, err := checkFreshness(answer, prompt)
		if err != nil {
			dbg("check-on-store %s failed: %v", cacheID, err)
			return
		}
		if res.Summary.Stale > 0 {
			store.update(cacheID, func(p *storedPrompt) { p.stale = true })
			err := deleteCacheEntry(cacheID)
			dbg("check-on-store %s: %d stale claim(s), deleted=%v", cacheID, res.Summary.Stale, err == nil)
			return
		}
		store.update(cacheID, func(p *storedPrompt) { p.checkedAt = timeNow() })
	}()
}

// deleteCacheEntry removes one semantic-cache entry through Bifrost's documented
// admin API (DELETE /api/cache/clear/{cacheId}).
func deleteCacheEntry(cacheID string) error {
	if cacheID == "" {
		return fmt.Errorf("no cache id")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	target := strings.TrimRight(cfg.LoopbackURL, "/") + "/api/cache/clear/" + url.PathEscape(cacheID)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, target, nil)
	if err != nil {
		return err
	}
	if adminAuth != "" {
		req.Header.Set("Authorization", adminAuth)
	}
	if setupToken != "" {
		req.Header.Set("X-Bifrost-Setup-Token", setupToken)
	}
	res, err := regenHTTP.Do(req)
	if err != nil {
		return err
	}
	res.Body.Close()
	if res.StatusCode/100 != 2 {
		return fmt.Errorf("cache delete status %d", res.StatusCode)
	}
	return nil
}
