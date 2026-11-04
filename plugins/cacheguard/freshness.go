package main

// Best-Before: time-sensitive cached answers get a review deadline. After it, a cache
// hit is checked claim by claim against live web sources (the freshness checker
// service, which uses Tavily) BEFORE it is served. If a claim is stale, the cache
// entry is deleted and the answer is regenerated with the fresh evidence added (by
// the small companion plugin "cacheguard-evidence"), so the new answer is built from
// current sources and is cached in place of the old one.
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

// checkResult mirrors the freshness checker's POST /check response (fields we use).
type checkResult struct {
	Claims []struct {
		Text       string `json:"text"`
		Flagged    bool   `json:"flagged"`
		Verdict    string `json:"verdict"`
		Label      string `json:"label"`
		Score      int    `json:"score"`
		NewerValue string `json:"newer_value"`
		Evidence   []struct {
			Line      string `json:"line"`
			URL       string `json:"url"`
			Published string `json:"published"`
		} `json:"evidence"`
	} `json:"claims"`
	Summary struct {
		Flagged  int  `json:"flagged"`
		Checked  int  `json:"checked"`
		Stale    int  `json:"stale"`
		MinScore *int `json:"min_score"`
	} `json:"summary"`
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
// when regenerating (only lines that came from the sources, never model text).
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
			fmt.Fprintf(&b, "- This older statement is out of date: %q.\n", c.Text)
		}
		for i, e := range c.Evidence {
			if i == 2 {
				break
			}
			date := e.Published
			if len(date) > 10 {
				date = date[:10]
			}
			fmt.Fprintf(&b, "  Source (%s, %s): %q\n", e.URL, date, e.Line)
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
	d.ClaimsChecked, d.StaleClaims = res.Summary.Checked, res.Summary.Stale
	if res.Summary.MinScore != nil {
		d.MinScore = *res.Summary.MinScore
	}
	if res.Summary.Stale > 0 {
		d.Action, d.Reason, d.Freshness = "reject", "stale-claim", "stale"
		d.Detail, d.Evidence = res.staleDetail()
		ev := d.Evidence
		store.update(d.CacheID, func(p *storedPrompt) { p.stale, p.evidence = true, ev })
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
		if res.Summary.Stale == 0 {
			store.update(cacheID, func(p *storedPrompt) { p.checkedAt = timeNow() })
			dbg("refresh-ahead %s: still fresh, deadline extended", cacheID)
			return
		}
		_, evidence := res.staleDetail()
		store.update(cacheID, func(p *storedPrompt) { p.stale, p.evidence = true, evidence })
		if raw == nil || cfg.Mode != "enforce" {
			return
		}
		keep := deleteCacheEntry(cacheID) == nil
		status, _, _, err := loopback(nil, raw, evidence, keep)
		dbg("refresh-ahead %s: stale, removed=%v, regenerated status=%d err=%v", cacheID, keep, status, err)
	}()
	return true
}

// checkOnStore checks a newly cached time-sensitive answer in the background. A stale
// answer is marked (so any hit regenerates) and deleted from the cache.
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
