// Cache Guard: a Bifrost plugin that checks every semantic-cache hit before it is
// served, and regenerates the answer when the cached entry was written for a
// question that only *looks* similar (with vs without, v2 vs v3, EU vs US ...).
//
// It runs on an unmodified, dynamically built Bifrost through the documented plugin
// interface. Register it with placement "pre_builtin" so it runs before semantic_cache:
//   - its PreLLMHook runs first, so it can restrict unsupported requests to exact-match
//     caching (no unchecked semantic replay);
//   - its PostLLMHook sees every cache hit (core replays post hooks for the plugins
//     that ran before a short-circuit) and decides whether the hit may be served.
//
// Supported surface (v1): non-streaming, single-turn text chat/text completions.
// Everything else is limited to exact-match caching for that request.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
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
	"github.com/maximhq/bifrost/plugins/cacheguard/guard"
	"github.com/maximhq/bifrost/plugins/semanticcache"
)

// Config is read from the plugin's "config" block.
type Config struct {
	Mode                string   `json:"mode"`                  // "enforce" (default): replace rejected hits; "shadow": headers only
	LoopbackURL         string   `json:"loopback_url"`          // this Bifrost's own address, default http://127.0.0.1:8080
	AllowRemoteLoopback bool     `json:"allow_remote_loopback"` // loopback_url must be localhost unless this is true
	RegenTimeoutMs      int      `json:"regen_timeout_ms"`      // limit for the one regeneration request
	DisableRules        []string `json:"disable_rules"`         // stage-1 rules to switch off, e.g. ["substitution"]
	Stage2URL           string   `json:"stage2_url"`            // optional cross-encoder service, e.g. http://127.0.0.1:8099/score
	Stage2Threshold     float64  `json:"stage2_threshold"`      // serve only if stage-2 score >= this
	Stage2TimeoutMs     int      `json:"stage2_timeout_ms"`     // timeout/error => reject (fail closed)
	MaxEntries          int      `json:"max_entries"`           // prompts remembered for cache entries (FIFO bound)
	OnUnknown           string   `json:"on_unknown"`            // hit on an entry whose question we never saw: "reject" (default) or "serve"

	BestBefore BestBeforeConfig `json:"best_before"` // freshness checks for time-sensitive answers (see freshness.go)
}

type decision struct {
	Action         string // "pass" | "reject" | "unverified" | "bypass"
	Reason         string
	Detail         string
	Stage          int
	HitType        string
	Similarity     float64
	Score          float64
	Latency        time.Duration
	Regenerated    string // "", "true", "failed", "skipped"
	RegenRequestID string
	CacheID        string

	// Best-Before
	Freshness        string // within-deadline | rechecked-fresh | stale | check-failed | not-time-sensitive
	FreshDeadline    time.Time
	ReviewEvery      time.Duration // review interval used for this answer
	ReviewBasis      string        // "ttl" (fixed per category) or "max-stale-rate"
	ClaimsChecked    int
	StaleClaims      int
	MinScore         int
	CheckLatency     time.Duration
	Evidence         string // evidence text for the regeneration (sources only)
	EvidenceInjected string // "injected" | "not-injected" (regeneration after a stale claim)
	CacheDelete      string // "ok" or the reason removing the old entry failed
	Refresh          string // "started": a background refresh-ahead check began
}

type rawRequest struct {
	Method  string
	Path    string
	Query   map[string]string
	Headers map[string]string
	Body    []byte
}

// reqState is created by the HTTP pre-hook and shared by pointer with the LLM hooks.
// Bifrost gives HTTP hooks and LLM hooks different context objects, but values set in
// the HTTP pre-hook are copied into the request, so the LLM hooks can read (and fill in)
// this holder, and the HTTP post-hooks see the result.
type reqState struct {
	mu       sync.Mutex
	raw      *rawRequest
	decision *decision
}

type storedPrompt struct {
	prompt    string
	system    string    // exact system/developer prefix the answer was generated with
	cachedAt  time.Time // when the answer was cached
	checkedAt time.Time // last time a freshness check found no stale claim
	stale     bool      // a freshness check found a stale claim: never serve
	evidence  string    // fresh source lines from that check, reused for every regeneration
}

const (
	pluginName   = "cacheguard"
	loopbackMark = "x-cacheguard-loopback"
	hdrPrefix    = "x-bifrost-cacheguard-"
)

var (
	keyPrompt = schemas.BifrostContextKey("cacheguard-prompt")
	keySystem = schemas.BifrostContextKey("cacheguard-system")
	keyState  = schemas.BifrostContextKey("cacheguard-state")
	keyRegen  = schemas.BifrostContextKey("cacheguard-is-regeneration") // bool: this request is our own regeneration

	// Secret marker for our own regeneration requests (random per process), so a client
	// cannot pretend to be a regeneration or inject evidence.
	loopbackToken string

	cfg       Config
	ruleOpts  guard.Options
	store     *promptStore
	regenHTTP *http.Client
	debug     = os.Getenv("CACHEGUARD_DEBUG") == "1"

	// Fallback channel between LLM hooks and HTTP hooks, keyed by Bifrost's request ID
	// (also returned to the client as x-bifrost-request-id).
	byRequestID sync.Map // string -> *timedDecision
)

type timedDecision struct {
	d  *decision
	at time.Time
}

func dbg(format string, args ...any) {
	if debug {
		fmt.Fprintf(os.Stderr, "[cacheguard] "+format+"\n", args...)
	}
}

// ---------------------------------------------------------------- lifecycle

func Init(config any) error {
	cfg = Config{Mode: "enforce", LoopbackURL: "http://127.0.0.1:8080", RegenTimeoutMs: 120_000,
		Stage2Threshold: 0.5, Stage2TimeoutMs: 150, MaxEntries: 100_000, OnUnknown: "reject"}
	if config != nil {
		b, err := json.Marshal(config)
		if err != nil {
			return fmt.Errorf("cacheguard: bad config: %w", err)
		}
		if err := json.Unmarshal(b, &cfg); err != nil {
			return fmt.Errorf("cacheguard: bad config: %w", err)
		}
	}
	if cfg.Mode != "enforce" && cfg.Mode != "shadow" {
		return fmt.Errorf("cacheguard: mode must be \"enforce\" or \"shadow\", got %q", cfg.Mode)
	}
	if cfg.OnUnknown != "reject" && cfg.OnUnknown != "serve" {
		return fmt.Errorf("cacheguard: on_unknown must be \"reject\" or \"serve\", got %q", cfg.OnUnknown)
	}
	u, err := url.Parse(cfg.LoopbackURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("cacheguard: loopback_url must be an http(s) URL, got %q", cfg.LoopbackURL)
	}
	if h := u.Hostname(); !cfg.AllowRemoteLoopback && h != "127.0.0.1" && h != "localhost" && h != "::1" {
		return fmt.Errorf("cacheguard: loopback_url must point at this Bifrost (localhost); set allow_remote_loopback to override")
	}
	if err := initBestBefore(); err != nil {
		return err
	}
	var tok [16]byte
	if _, err := rand.Read(tok[:]); err != nil {
		return fmt.Errorf("cacheguard: %w", err)
	}
	loopbackToken = hex.EncodeToString(tok[:])
	ruleOpts = guard.Options{Disabled: map[string]bool{}}
	for _, r := range cfg.DisableRules {
		ruleOpts.Disabled[r] = true
	}
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 100_000
	}
	store = newPromptStore(cfg.MaxEntries)
	regenHTTP = &http.Client{
		Timeout: time.Duration(cfg.RegenTimeoutMs) * time.Millisecond,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse // never follow redirects off this gateway
		},
	}
	return nil
}

func GetName() string { return pluginName }

func Cleanup() error { return nil }

// ---------------------------------------------------------------- hooks

func HTTPTransportPreHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	if req == nil || !strings.EqualFold(req.Method, http.MethodPost) {
		return nil, nil
	}
	ctx.SetValue(keyState, &reqState{raw: &rawRequest{
		Method:  req.Method,
		Path:    req.Path,
		Query:   copyMap(req.Query),
		Headers: copyMap(req.Headers),
		Body:    append([]byte(nil), req.Body...),
	}})
	dbg("http pre-hook %s %s", req.Method, req.Path)
	return nil, nil
}

func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	if req == nil {
		return req, nil, nil
	}
	if st, ok := ctx.Value(keyState).(*reqState); ok && st.raw != nil && loopbackToken != "" &&
		headerValue(st.raw.Headers, loopbackMark) == loopbackToken {
		ctx.SetValue(keyRegen, true)
		if ev := headerValue(st.raw.Headers, evidenceHeader); ev != "" {
			if b, err := base64.StdEncoding.DecodeString(ev); err == nil && len(b) > 0 {
				ctx.SetValue(evidenceTextKey, string(b))
				ctx.SetValue(evidenceDoneKey, new(atomic.Bool))
			}
		}
	}
	prompt, system, why := analyse(req)
	if why != "" {
		// Not in the supported surface: allow exact-match caching only for this request,
		// so no unchecked semantic replay can happen.
		ctx.SetValue(semanticcache.CacheTypeKey, semanticcache.CacheTypeDirect)
		publish(ctx, &decision{Action: "bypass", Reason: why})
		dbg("pre-llm: %s -> semantic lookup disabled for this request", why)
		return req, nil, nil
	}
	ctx.SetValue(keyPrompt, prompt)
	ctx.SetValue(keySystem, system)
	dbg("pre-llm prompt=%q", prompt)
	return req, nil, nil
}

func PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if resp == nil {
		return resp, bifrostErr, nil
	}
	ef := resp.GetExtraFields()
	if ef == nil || ef.CacheDebug == nil || ef.CacheDebug.CacheID == nil {
		// Cache did not run for this request. A regeneration sent without a cache key
		// still reports whether the evidence was used.
		if regen, _ := ctx.Value(keyRegen).(bool); regen {
			if done, ok := ctx.Value(evidenceDoneKey).(*atomic.Bool); ok {
				injected := "not-injected"
				if done.Load() {
					injected = "injected"
				}
				publish(ctx, &decision{Action: "generated", EvidenceInjected: injected})
			}
		}
		return resp, bifrostErr, nil
	}
	cd := ef.CacheDebug
	prompt, _ := ctx.Value(keyPrompt).(string)
	system, _ := ctx.Value(keySystem).(string)

	if !cd.CacheHit {
		// The semantic cache will store this answer under CacheID: remember its question.
		if prompt != "" {
			now := timeNow()
			entry := storedPrompt{prompt: prompt, system: system, cachedAt: now}
			injected := ""
			if done, ok := ctx.Value(evidenceDoneKey).(*atomic.Bool); ok {
				injected = "not-injected"
				if done.Load() {
					injected, entry.checkedAt = "injected", now // built from sources fetched just now
				}
			}
			store.put(*cd.CacheID, entry)
			if regen, _ := ctx.Value(keyRegen).(bool); regen {
				publish(ctx, &decision{Action: "generated", EvidenceInjected: injected})
			}
			if injected != "injected" && bifrostErr == nil {
				checkOnStore(*cd.CacheID, prompt, answerText(resp))
			}
		}
		dbg("post-llm MISS cache_id=%s remembered=%v", *cd.CacheID, prompt != "")
		return resp, bifrostErr, nil
	}

	start := time.Now()
	d := &decision{Action: "pass", HitType: deref(cd.HitType), CacheID: *cd.CacheID, MinScore: -1}
	if cd.Similarity != nil {
		d.Similarity = *cd.Similarity
	}
	cached, known := store.get(*cd.CacheID)
	switch {
	case d.HitType == "direct":
		d.Reason = "exact-match"
	case prompt == "":
		// Request outside the supported surface somehow reached a semantic hit.
		d.Action, d.Reason = "reject", "unsupported-request"
	case !known:
		// We never saw the question this entry was written for (e.g. after a restart).
		d.Action, d.Reason = "reject", "unknown-entry"
		if cfg.OnUnknown == "serve" {
			d.Action = "unverified"
		}
	case cached.system != system:
		d.Action, d.Reason, d.Stage = "reject", "system-prompt-differs", 1
	default:
		v := guard.VerifyWith(prompt, cached.prompt, ruleOpts)
		d.Stage = 1
		if !v.Reuse {
			d.Action, d.Reason, d.Detail = "reject", v.Reason, v.Detail
		} else if cfg.Stage2URL != "" {
			d.Stage = 2
			score, err := stage2Score(prompt, cached.prompt)
			d.Score = score
			switch {
			case err != nil:
				d.Action, d.Reason, d.Detail = "reject", "stage2-unavailable", err.Error()
			case score < cfg.Stage2Threshold:
				d.Action, d.Reason = "reject", "stage2-low-score"
			}
		}
	}
	d.Latency = time.Since(start)
	// Our own regeneration request never re-runs the freshness check (it would recurse);
	// it can only hit an identical question, and it is sent without the cache key when
	// the old entry could not be removed.
	if regen, _ := ctx.Value(keyRegen).(bool); !regen && (d.Action == "pass" || d.Action == "unverified") {
		var raw *rawRequest
		if st, ok := ctx.Value(keyState).(*reqState); ok {
			raw = st.raw
		}
		bestBeforeOnHit(d, resp, prompt, cached, known, raw)
	}
	publish(ctx, d)
	ctx.Log(schemas.LogLevelInfo, fmt.Sprintf("cacheguard: %s hit -> %s (%s %s) in %s", d.HitType, d.Action, d.Reason, d.Detail, d.Latency))
	dbg("post-llm HIT %s cache_id=%s -> %s %s %s", d.HitType, *cd.CacheID, d.Action, d.Reason, d.Detail)
	return resp, bifrostErr, nil
}

func HTTPTransportPostHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	if resp == nil {
		return nil
	}
	d := lookup(ctx, resp.Headers, false)
	if d == nil {
		return nil
	}
	if resp.Headers == nil {
		resp.Headers = map[string]string{}
	}
	if d.Action == "reject" && cfg.Mode == "enforce" && d.Regenerated == "" {
		regenerate(ctx, req, resp, d)
	}
	for k, v := range d.headers() {
		resp.Headers[k] = v
	}
	return nil
}

// HTTPTransportResponseHeadersHook is the last point before headers are committed.
// It makes the cache headers describe the answer actually served.
func HTTPTransportResponseHeadersHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponseMetadata) error {
	if resp == nil {
		return nil
	}
	d := lookup(ctx, resp.Headers, false)
	if d == nil {
		return nil
	}
	for k, v := range d.headers() {
		resp.SetHeader(k, v)
	}
	return nil
}

// regenerate replaces a rejected cached answer with a freshly generated one by
// re-sending the original request to this Bifrost with semantic lookup disabled.
// Exactly one attempt; a request that is itself a regeneration is never regenerated again.
func regenerate(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse, d *decision) {
	var raw *rawRequest
	if st, ok := ctx.Value(keyState).(*reqState); ok {
		raw = st.raw
	}
	if raw == nil && req != nil && len(req.Body) > 0 {
		raw = &rawRequest{Method: req.Method, Path: req.Path, Query: copyMap(req.Query), Headers: copyMap(req.Headers), Body: append([]byte(nil), req.Body...)}
	}
	fail := func(why string) {
		// Fail closed: never serve an answer we just judged to be for another question.
		resp.StatusCode = http.StatusBadGateway
		resp.Body = []byte(fmt.Sprintf(`{"error":{"type":"cacheguard_regeneration_failed","message":"cached answer rejected (%s) and regeneration failed: %s"}}`, jsonEscape(d.Reason), jsonEscape(why)))
		resp.Headers["content-type"] = "application/json"
		d.Regenerated = "failed"
	}
	switch {
	case raw == nil:
		fail("original request not available")
		return
	case loopbackToken != "" && headerValue(raw.Headers, loopbackMark) == loopbackToken:
		fail("request was already a regeneration")
		return
	}
	keepCacheKey := true
	if d.Reason == "stale-claim" || d.Reason == "known-stale" || d.Reason == "freshness-check-failed" {
		// The regeneration has the same question, so it would hit this same entry again.
		// Remove the entry first, so the regenerated answer is cached in its place. If that
		// fails, regenerate without the cache so the old entry cannot be hit again.
		if err := deleteCacheEntry(d.CacheID); err != nil {
			keepCacheKey = false
			d.CacheDelete = "failed: " + err.Error()
			dbg("could not delete stale entry %s: %v", d.CacheID, err)
		} else {
			d.CacheDelete = "ok"
		}
	}
	status, headers, body, err := loopback(ctx, raw, d.Evidence, keepCacheKey)
	if err != nil {
		fail(err.Error())
		return
	}
	resp.StatusCode = status
	resp.Body = body
	if ct := headerValue(headers, "content-type"); ct != "" {
		resp.Headers["content-type"] = ct
	}
	d.RegenRequestID = headerValue(headers, "x-bifrost-request-id")
	d.Regenerated = "true"
	if d.Evidence != "" {
		d.EvidenceInjected = headerValue(headers, hdrPrefix+"evidence")
		if d.EvidenceInjected == "" {
			d.EvidenceInjected = "not-injected"
		}
	}
}

// ---------------------------------------------------------------- helpers

func (d *decision) headers() map[string]string {
	h := map[string]string{
		hdrPrefix + "decision": d.Action,
		hdrPrefix + "mode":     cfg.Mode,
	}
	if d.Action == "generated" { // our own regeneration request: only report evidence use
		if d.EvidenceInjected != "" {
			h[hdrPrefix+"evidence"] = d.EvidenceInjected
		}
		return h
	}
	if d.Freshness != "" {
		h[hdrPrefix+"freshness"] = d.Freshness
		if !d.FreshDeadline.IsZero() {
			h[hdrPrefix+"freshness-deadline"] = d.FreshDeadline.UTC().Format(time.RFC3339)
		}
		if d.ReviewEvery > 0 {
			h[hdrPrefix+"review-every"] = fmt.Sprintf("%s (%s)", d.ReviewEvery.Round(time.Minute), d.ReviewBasis)
		}
		if d.ClaimsChecked > 0 || d.StaleClaims > 0 {
			h[hdrPrefix+"claims-checked"] = fmt.Sprint(d.ClaimsChecked)
			h[hdrPrefix+"stale-claims"] = fmt.Sprint(d.StaleClaims)
			h[hdrPrefix+"freshness-check-ms"] = fmt.Sprint(d.CheckLatency.Milliseconds())
		}
		if d.MinScore >= 0 {
			h[hdrPrefix+"min-freshness-score"] = fmt.Sprint(d.MinScore)
		}
	}
	if d.EvidenceInjected != "" {
		h[hdrPrefix+"evidence"] = d.EvidenceInjected
	}
	if d.Refresh != "" {
		h[hdrPrefix+"refresh"] = d.Refresh
	}
	if d.CacheDelete != "" {
		h[hdrPrefix+"old-entry-removed"] = headerSafe(d.CacheDelete)
	}
	if d.Reason != "" {
		h[hdrPrefix+"reason"] = d.Reason
	}
	if d.Detail != "" {
		h[hdrPrefix+"detail"] = headerSafe(d.Detail)
	}
	if d.Stage > 0 {
		h[hdrPrefix+"stage"] = fmt.Sprint(d.Stage)
	}
	if d.Stage == 2 {
		h[hdrPrefix+"score"] = fmt.Sprintf("%.4f", d.Score)
	}
	if d.Action != "bypass" && d.HitType != "" {
		h[hdrPrefix+"decision-latency-us"] = fmt.Sprint(d.Latency.Microseconds())
		h[hdrPrefix+"candidate-hit-type"] = d.HitType
		if d.HitType == "semantic" {
			h[hdrPrefix+"candidate-similarity"] = fmt.Sprintf("%.4f", d.Similarity)
		}
	}
	if d.Action == "reject" && cfg.Mode == "enforce" {
		// The candidate was not served, so the response is not a cache hit.
		h["x-bifrost-cache-hit"] = "false"
		h["x-bifrost-cache-hit-type"] = "none"
		if d.Regenerated != "" {
			h[hdrPrefix+"regenerated"] = d.Regenerated
		}
		if d.RegenRequestID != "" {
			h[hdrPrefix+"regeneration-request-id"] = d.RegenRequestID
		}
	}
	return h
}

// analyse returns the user prompt and system prefix for supported requests, or a
// reason why the request is outside the supported surface.
func analyse(req *schemas.BifrostRequest) (prompt, system, why string) {
	switch req.RequestType {
	case schemas.ChatCompletionRequest:
		cr := req.ChatRequest
		if cr == nil {
			return "", "", "no-chat-input"
		}
		if cr.Params != nil && len(cr.Params.Tools) > 0 {
			return "", "", "tools"
		}
		var users, systems []string
		for _, m := range cr.Input {
			text, ok := messageText(m)
			if !ok {
				return "", "", "non-text-content"
			}
			switch m.Role {
			case schemas.ChatMessageRoleUser:
				users = append(users, text)
			case schemas.ChatMessageRoleSystem, schemas.ChatMessageRoleDeveloper:
				systems = append(systems, text)
			default:
				return "", "", "multi-turn"
			}
		}
		if len(users) != 1 {
			return "", "", "multi-turn"
		}
		return users[0], strings.Join(systems, "\n"), ""
	case schemas.TextCompletionRequest:
		tr := req.TextCompletionRequest
		if tr == nil || tr.Input == nil || tr.Input.PromptStr == nil {
			return "", "", "non-text-content"
		}
		return *tr.Input.PromptStr, "", ""
	default:
		if strings.HasSuffix(string(req.RequestType), "_stream") {
			return "", "", "streaming"
		}
		return "", "", "unsupported-type:" + string(req.RequestType)
	}
}

func messageText(m schemas.ChatMessage) (string, bool) {
	if m.Content == nil {
		return "", true
	}
	if m.Content.ContentStr != nil {
		return *m.Content.ContentStr, true
	}
	var parts []string
	for _, b := range m.Content.ContentBlocks {
		if b.Type != schemas.ChatContentBlockTypeText || b.Text == nil {
			return "", false
		}
		parts = append(parts, *b.Text)
	}
	return strings.Join(parts, "\n"), true
}

// Headers forwarded to the regeneration request: identity/governance and cache scoping only.
func forwardHeader(k string) bool {
	k = strings.ToLower(k)
	switch k {
	case "authorization", "x-api-key", "content-type", "accept":
		return true
	case "x-bf-cache-type", "x-bf-cache-threshold", loopbackMark, evidenceHeader:
		return false
	}
	return strings.HasPrefix(k, "x-bf-")
}

// loopback re-sends the original request to this Bifrost. Semantic matching is
// effectively off (threshold 1.0: only an identical question matches), but the new
// answer is cached normally under the same cache key, embeddings included.
// evidence (optional) is added to the model input by the cacheguard-evidence plugin.
// keepCacheKey=false sends the request without the cache key (no lookup, no store).
func loopback(ctx *schemas.BifrostContext, raw *rawRequest, evidence string, keepCacheKey bool) (int, map[string]string, []byte, error) {
	if !strings.HasPrefix(raw.Path, "/") {
		return 0, nil, nil, errors.New("invalid request path")
	}
	target := strings.TrimRight(cfg.LoopbackURL, "/") + raw.Path
	if len(raw.Query) > 0 {
		q := url.Values{}
		for k, v := range raw.Query {
			q.Set(k, v)
		}
		target += "?" + q.Encode()
	}
	var parent context.Context = context.Background()
	if ctx != nil {
		parent = ctx
	}
	req, err := http.NewRequestWithContext(parent, http.MethodPost, target, bytes.NewReader(raw.Body))
	if err != nil {
		return 0, nil, nil, err
	}
	for k, v := range raw.Headers {
		if forwardHeader(k) {
			req.Header.Set(k, v)
		}
	}
	req.Header.Set("x-bf-cache-threshold", "1") // only an identical question can match
	if !keepCacheKey {
		req.Header.Del("x-bf-cache-key")
	}
	if evidence != "" {
		req.Header.Set(evidenceHeader, base64.StdEncoding.EncodeToString([]byte(evidence)))
	}
	req.Header.Set(loopbackMark, loopbackToken)
	res, err := regenHTTP.Do(req)
	if err != nil {
		return 0, nil, nil, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
	if err != nil {
		return 0, nil, nil, err
	}
	headers := map[string]string{}
	for k := range res.Header {
		headers[strings.ToLower(k)] = res.Header.Get(k)
	}
	return res.StatusCode, headers, body, nil
}

// stage2Score asks the optional cross-encoder service whether a and b ask the same thing.
func stage2Score(a, b string) (float64, error) {
	payload, _ := json.Marshal(map[string]string{"a": a, "b": b})
	c := &http.Client{Timeout: time.Duration(cfg.Stage2TimeoutMs) * time.Millisecond}
	res, err := c.Post(cfg.Stage2URL, "application/json", bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("stage2 status %d", res.StatusCode)
	}
	var out struct {
		Score *float64 `json:"score"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, err
	}
	if out.Score == nil || *out.Score != *out.Score { // missing or NaN
		return 0, errors.New("stage2 returned no valid score")
	}
	return *out.Score, nil
}

// publish makes a decision visible to the HTTP hooks (shared holder + request-ID map).
func publish(ctx *schemas.BifrostContext, d *decision) {
	if st, ok := ctx.Value(keyState).(*reqState); ok {
		st.mu.Lock()
		st.decision = d
		st.mu.Unlock()
	}
	if rid, ok := ctx.Value(schemas.BifrostContextKeyRequestID).(string); ok && rid != "" {
		byRequestID.Store(rid, &timedDecision{d: d, at: time.Now()})
	}
	sweep()
}

// lookup finds the decision for this request from the HTTP side.
func lookup(ctx *schemas.BifrostContext, headers map[string]string, consume bool) *decision {
	if st, ok := ctx.Value(keyState).(*reqState); ok {
		st.mu.Lock()
		d := st.decision
		st.mu.Unlock()
		if d != nil {
			return d
		}
	}
	rid := headerValue(headers, "x-bifrost-request-id")
	if rid == "" {
		return nil
	}
	var v any
	var ok bool
	if consume {
		v, ok = byRequestID.LoadAndDelete(rid)
	} else {
		v, ok = byRequestID.Load(rid)
	}
	if !ok {
		return nil
	}
	return v.(*timedDecision).d
}

var (
	lastSweep time.Time
	sweepMu   sync.Mutex
)

// sweep drops request-ID entries older than two minutes.
func sweep() {
	sweepMu.Lock()
	defer sweepMu.Unlock()
	if time.Since(lastSweep) < 30*time.Second {
		return
	}
	lastSweep = time.Now()
	byRequestID.Range(func(k, v any) bool {
		if time.Since(v.(*timedDecision).at) > 2*time.Minute {
			byRequestID.Delete(k)
		}
		return true
	})
}

// promptStore remembers which question each cache entry was written for.
// Bounded FIFO: the oldest entries are forgotten first (their hits then count as unknown).
type promptStore struct {
	mu    sync.RWMutex
	max   int
	m     map[string]storedPrompt
	order []string
}

func newPromptStore(max int) *promptStore {
	return &promptStore{max: max, m: make(map[string]storedPrompt)}
}

func (s *promptStore) put(id string, p storedPrompt) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.m[id]; !ok {
		s.order = append(s.order, id)
	}
	s.m[id] = p
	for len(s.order) > s.max {
		delete(s.m, s.order[0])
		s.order = s.order[1:]
	}
}

func (s *promptStore) update(id string, f func(*storedPrompt)) {
	if s == nil || id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if p, ok := s.m[id]; ok {
		f(&p)
		s.m[id] = p
	}
}

func (s *promptStore) get(id string) (storedPrompt, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.m[id]
	return p, ok
}

func headerValue(h map[string]string, key string) string {
	for k, v := range h {
		if strings.EqualFold(k, key) {
			return v
		}
	}
	return ""
}

func copyMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func headerSafe(s string) string {
	s = strings.Map(func(r rune) rune {
		if r < 0x20 || r > 0x7e {
			return '?'
		}
		return r
	}, s)
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}
