// Cache Guard: a Bifrost plugin that checks every semantic-cache hit before it is
// served, and regenerates the answer when the cached entry was written for a
// question that only *looks* similar (with vs without, v2 vs v3, EU vs US ...).
//
// It runs on a stock Bifrost build through the documented dynamic-plugin interface
// and needs no change to Bifrost's own code. Register it with placement "pre_builtin"
// so it runs before the semantic_cache plugin: its PostLLMHook then sees every
// cache hit (core replays post hooks for the plugins that ran before a short-circuit).
//
// Flow per request:
//
//	HTTPTransportPreHook  keep a copy of the raw HTTP request (for regeneration)
//	PreLLMHook            remember the user's prompt text
//	PostLLMHook           cache MISS -> remember "cache entry ID -> prompt"
//	                      cache HIT  -> stage 1 rules (+ optional stage 2 model)
//	HTTPTransportPostHook reject -> re-ask Bifrost with x-bf-cache-type: direct
//	                      and replace the response body; always add decision headers
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/plugins/cacheguard/guard"
)

// Config is read from the plugin's "config" block.
type Config struct {
	Mode            string  `json:"mode"`              // "enforce" (default): replace rejected hits; "shadow": headers only
	LoopbackURL     string  `json:"loopback_url"`      // this Bifrost's own address, default http://127.0.0.1:8080
	Stage2URL       string  `json:"stage2_url"`        // optional cross-encoder service, e.g. http://127.0.0.1:8099/score
	Stage2Threshold float64 `json:"stage2_threshold"`  // serve only if stage-2 score >= this
	Stage2TimeoutMs int     `json:"stage2_timeout_ms"` // fail closed (reject) when exceeded
	MaxEntries      int     `json:"max_entries"`       // prompts remembered for cache entries
	OnUnknown       string  `json:"on_unknown"`        // hit on an entry we never saw written: "serve" (default) or "reject"
}

type decision struct {
	Action     string // "pass" | "reject" | "unverified"
	Reason     string
	Detail     string
	Stage      int
	HitType    string
	Similarity float64
	Score      float64
	Latency    time.Duration
}

type rawRequest struct {
	Method  string
	Path    string
	Query   map[string]string
	Headers map[string]string
	Body    []byte
}

const (
	pluginName   = "cacheguard"
	bypassHeader = "x-cacheguard-loopback"
	hdrPrefix    = "x-bifrost-cacheguard-"
)

// reqState is created by the HTTP pre-hook and shared by pointer with the LLM hooks.
// Bifrost gives HTTP hooks and LLM hooks different context objects, but values set
// in the HTTP pre-hook are copied into the request, so the LLM hooks can read (and
// fill in) this holder, and the HTTP post-hooks see the result.
type reqState struct {
	mu       sync.Mutex
	raw      *rawRequest
	decision *decision
}

var (
	keyPrompt = schemas.BifrostContextKey("cacheguard-prompt")
	keyState  = schemas.BifrostContextKey("cacheguard-state")

	cfg    Config
	store  *promptStore
	client *http.Client
	debug  = os.Getenv("CACHEGUARD_DEBUG") == "1"

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
	cfg = Config{Mode: "enforce", LoopbackURL: "http://127.0.0.1:8080", Stage2Threshold: 0.5,
		Stage2TimeoutMs: 150, MaxEntries: 100_000, OnUnknown: "serve"}
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
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 100_000
	}
	store = newPromptStore(cfg.MaxEntries)
	client = &http.Client{Timeout: 120 * time.Second}
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
	if p := promptText(req); p != "" {
		ctx.SetValue(keyPrompt, p)
		dbg("pre-llm prompt=%q", p)
	}
	return req, nil, nil
}

func PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if resp == nil {
		return resp, bifrostErr, nil
	}
	ef := resp.GetExtraFields()
	if ef == nil || ef.CacheDebug == nil || ef.CacheDebug.CacheID == nil {
		dbg("post-llm: no cache metadata on this response")
		return resp, bifrostErr, nil // cache did not run for this request
	}
	cd := ef.CacheDebug
	prompt, _ := ctx.Value(keyPrompt).(string)

	if !cd.CacheHit {
		// The semantic cache will store this answer under CacheID: remember its question.
		if prompt != "" {
			store.put(*cd.CacheID, prompt)
		}
		dbg("post-llm MISS cache_id=%s remembered=%v", *cd.CacheID, prompt != "")
		return resp, bifrostErr, nil
	}

	start := time.Now()
	d := &decision{Action: "pass", HitType: deref(cd.HitType)}
	if cd.Similarity != nil {
		d.Similarity = *cd.Similarity
	}
	cached, known := store.get(*cd.CacheID)
	switch {
	case d.HitType == "direct":
		d.Reason = "exact-match"
	case !known || prompt == "":
		d.Action, d.Reason = "unverified", "unknown-entry"
		if cfg.OnUnknown == "reject" {
			d.Action = "reject"
		}
	default:
		v := guard.Verify(prompt, cached)
		d.Stage = 1
		if !v.Reuse {
			d.Action, d.Reason, d.Detail = "reject", v.Reason, v.Detail
		} else if cfg.Stage2URL != "" {
			d.Stage = 2
			score, err := stage2Score(prompt, cached)
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
	publish(ctx, d)
	dbg("post-llm HIT %s cache_id=%s -> %s %s %s", d.HitType, *cd.CacheID, d.Action, d.Reason, d.Detail)
	ctx.Log(schemas.LogLevelInfo, fmt.Sprintf("cacheguard: %s hit -> %s (%s %s) in %s", d.HitType, d.Action, d.Reason, d.Detail, d.Latency))
	return resp, bifrostErr, nil
}

func HTTPTransportResponseHeadersHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponseMetadata) error {
	if resp == nil {
		return nil
	}
	if d := lookup(ctx, resp.Headers, false); d != nil {
		for k, v := range d.headers() {
			resp.SetHeader(k, v)
		}
	}
	return nil
}

func HTTPTransportPostHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	if resp == nil {
		return nil
	}
	d := lookup(ctx, resp.Headers, true)
	dbg("http post-hook decision=%v", d != nil)
	if d == nil {
		return nil
	}
	if resp.Headers == nil {
		resp.Headers = map[string]string{}
	}
	for k, v := range d.headers() {
		resp.Headers[k] = v
	}
	if d.Action != "reject" || cfg.Mode != "enforce" {
		return nil
	}
	var raw *rawRequest
	if st, ok := ctx.Value(keyState).(*reqState); ok {
		raw = st.raw
	}
	if raw == nil && req != nil && len(req.Body) > 0 {
		raw = &rawRequest{Method: req.Method, Path: req.Path, Query: copyMap(req.Query), Headers: copyMap(req.Headers), Body: append([]byte(nil), req.Body...)}
	}
	if raw == nil {
		resp.Headers[hdrPrefix+"regenerated"] = "false-no-request"
		return nil
	}
	status, headers, body, err := loopback(raw)
	if err != nil {
		// Fail closed: never serve an answer we just judged to be for another question.
		resp.StatusCode = http.StatusBadGateway
		resp.Body = []byte(fmt.Sprintf(`{"error":{"message":"cacheguard: cached answer rejected and regeneration failed: %s"}}`, jsonEscape(err.Error())))
		resp.Headers["content-type"] = "application/json"
		resp.Headers[hdrPrefix+"regenerated"] = "failed"
		return nil
	}
	resp.StatusCode = status
	resp.Body = body
	for k, v := range headers {
		lk := strings.ToLower(k)
		if lk == "content-type" || strings.HasPrefix(lk, "x-bifrost-cache-") {
			resp.Headers[lk] = v
		}
	}
	resp.Headers[hdrPrefix+"regenerated"] = "true"
	return nil
}

// ---------------------------------------------------------------- helpers

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
	rid := ""
	for k, v := range headers {
		if strings.EqualFold(k, "x-bifrost-request-id") {
			rid = v
		}
	}
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

var lastSweep time.Time
var sweepMu sync.Mutex

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

func (d *decision) headers() map[string]string {
	h := map[string]string{
		hdrPrefix + "decision": d.Action,
		hdrPrefix + "mode":     cfg.Mode,
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
	h[hdrPrefix+"latency-us"] = fmt.Sprint(d.Latency.Microseconds())
	return h
}

// promptText returns the user text of a chat or text-completion request.
func promptText(req *schemas.BifrostRequest) string {
	if req == nil {
		return ""
	}
	if req.ChatRequest != nil {
		var parts []string
		for _, m := range req.ChatRequest.Input {
			if m.Role != schemas.ChatMessageRoleUser || m.Content == nil {
				continue
			}
			if m.Content.ContentStr != nil {
				parts = append(parts, *m.Content.ContentStr)
			}
			for _, b := range m.Content.ContentBlocks {
				if b.Text != nil {
					parts = append(parts, *b.Text)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	if req.TextCompletionRequest != nil && req.TextCompletionRequest.Input != nil && req.TextCompletionRequest.Input.PromptStr != nil {
		return *req.TextCompletionRequest.Input.PromptStr
	}
	return ""
}

// loopback re-sends the original request to this Bifrost with semantic lookup
// disabled, so the client gets a freshly generated answer for its own question.
func loopback(raw *rawRequest) (int, map[string]string, []byte, error) {
	url := strings.TrimRight(cfg.LoopbackURL, "/") + raw.Path
	if len(raw.Query) > 0 {
		var q []string
		for k, v := range raw.Query {
			q = append(q, k+"="+v)
		}
		url += "?" + strings.Join(q, "&")
	}
	req, err := http.NewRequest(raw.Method, url, bytes.NewReader(raw.Body))
	if err != nil {
		return 0, nil, nil, err
	}
	for k, v := range raw.Headers {
		switch strings.ToLower(k) {
		case "content-length", "host", "accept-encoding", "connection", "x-bf-cache-type":
			continue
		}
		req.Header.Set(k, v)
	}
	req.Header.Set("x-bf-cache-type", "direct")
	req.Header.Set(bypassHeader, "1")
	res, err := client.Do(req)
	if err != nil {
		return 0, nil, nil, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return 0, nil, nil, err
	}
	headers := map[string]string{}
	for k := range res.Header {
		headers[k] = res.Header.Get(k)
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
	var out struct {
		Score float64 `json:"score"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, err
	}
	return out.Score, nil
}

// promptStore remembers which question each cache entry was written for.
// Bounded FIFO: the oldest entries are forgotten first (their hits become "unverified").
type promptStore struct {
	mu    sync.RWMutex
	max   int
	m     map[string]string
	order []string
}

func newPromptStore(max int) *promptStore {
	return &promptStore{max: max, m: make(map[string]string)}
}

func (s *promptStore) put(id, prompt string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.m[id]; !ok {
		s.order = append(s.order, id)
	}
	s.m[id] = prompt
	for len(s.order) > s.max {
		delete(s.m, s.order[0])
		s.order = s.order[1:]
	}
}

func (s *promptStore) get(id string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.m[id]
	return p, ok
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
