package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func str(s string) *string { return &s }

func userMsg(text string) schemas.ChatMessage {
	return schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: str(text)}}
}

func mustInit(t *testing.T, conf map[string]any) {
	t.Helper()
	if err := Init(conf); err != nil {
		t.Fatalf("Init: %v", err)
	}
}

func TestInitValidation(t *testing.T) {
	bad := []map[string]any{
		{"mode": "loud"},
		{"on_unknown": "maybe"},
		{"loopback_url": "ftp://127.0.0.1:8080"},
		{"loopback_url": "http://evil.example.com"},
	}
	for _, c := range bad {
		if err := Init(c); err == nil {
			t.Errorf("Init(%v) should fail", c)
		}
	}
	mustInit(t, map[string]any{"loopback_url": "http://evil.example.com", "allow_remote_loopback": true})
	mustInit(t, nil)
	if cfg.OnUnknown != "reject" {
		t.Errorf("default on_unknown = %q, want reject", cfg.OnUnknown)
	}
	mustInit(t, map[string]any{"disable_rules": []string{"substitution"}})
	if !ruleOpts.Disabled["substitution"] {
		t.Error("disable_rules not applied")
	}
}

func TestAnalyse(t *testing.T) {
	sys := schemas.ChatMessage{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentStr: str("Be brief.")}}
	asst := schemas.ChatMessage{Role: "assistant", Content: &schemas.ChatMessageContent{ContentStr: str("Hi")}}
	img := schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{
		ContentBlocks: []schemas.ChatContentBlock{{Type: "image_url"}}}}
	chat := func(msgs ...schemas.ChatMessage) *schemas.BifrostRequest {
		return &schemas.BifrostRequest{RequestType: schemas.ChatCompletionRequest, ChatRequest: &schemas.BifrostChatRequest{Input: msgs}}
	}
	withTools := chat(userMsg("weather?"))
	withTools.ChatRequest.Params = &schemas.ChatParameters{Tools: []schemas.ChatTool{{}}}

	cases := []struct {
		name                string
		req                 *schemas.BifrostRequest
		prompt, system, why string
	}{
		{"single turn", chat(userMsg("What is Go?")), "What is Go?", "", ""},
		{"system kept", chat(sys, userMsg("What is Go?")), "What is Go?", "Be brief.", ""},
		{"multi turn", chat(userMsg("a"), asst, userMsg("b")), "", "", "multi-turn"},
		{"two users", chat(userMsg("a"), userMsg("b")), "", "", "multi-turn"},
		{"image", chat(img), "", "", "non-text-content"},
		{"tools", withTools, "", "", "tools"},
		{"stream", &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest}, "", "", "streaming"},
		{"other", &schemas.BifrostRequest{RequestType: "responses"}, "", "", "unsupported-type:responses"},
	}
	for _, c := range cases {
		p, s, why := analyse(c.req)
		if p != c.prompt || s != c.system || why != c.why {
			t.Errorf("%s: got (%q,%q,%q) want (%q,%q,%q)", c.name, p, s, why, c.prompt, c.system, c.why)
		}
	}
}

func TestForwardHeader(t *testing.T) {
	keep := []string{"Authorization", "x-api-key", "Content-Type", "x-bf-vk", "x-bf-cache-key"}
	drop := []string{"x-bf-cache-type", "x-cacheguard-loopback", "cookie", "host", "x-forwarded-for", "accept-encoding"}
	for _, h := range keep {
		if !forwardHeader(h) {
			t.Errorf("%s should be forwarded", h)
		}
	}
	for _, h := range drop {
		if forwardHeader(h) {
			t.Errorf("%s should not be forwarded", h)
		}
	}
}

func rejected() *decision {
	return &decision{Action: "reject", Reason: "negation", HitType: "semantic", Similarity: 0.93}
}

func TestRegenerateReplacesAnswerAndHeadersAreTruthful(t *testing.T) {
	var got *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r
		w.Header().Set("content-type", "application/json")
		w.Header().Set("x-bifrost-request-id", "regen-1")
		w.Write([]byte(`{"fresh":true}`))
	}))
	defer srv.Close()
	mustInit(t, map[string]any{"loopback_url": srv.URL})

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(`{}`),
		Headers: map[string]string{"Authorization": "Bearer k", "Cookie": "secret", "x-bf-cache-type": "semantic"}}
	resp := &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{}, Body: []byte(`{"stale":true}`)}
	d := rejected()
	regenerate(ctx, req, resp, d)

	if string(resp.Body) != `{"fresh":true}` || d.Regenerated != "true" || d.RegenRequestID != "regen-1" {
		t.Fatalf("not regenerated: body=%s d=%+v", resp.Body, d)
	}
	if got.Header.Get("x-bf-cache-type") != "direct" || got.Header.Get(loopbackMark) != "1" {
		t.Error("loopback must be direct-only and marked")
	}
	if got.Header.Get("Cookie") != "" || got.Header.Get("Authorization") != "Bearer k" {
		t.Error("header allowlist not applied")
	}
	h := d.headers()
	if h["x-bifrost-cache-hit"] != "false" || h[hdrPrefix+"candidate-hit-type"] != "semantic" || h[hdrPrefix+"regenerated"] != "true" {
		t.Errorf("headers not truthful: %v", h)
	}
}

func TestShadowModeKeepsCacheHeaders(t *testing.T) {
	mustInit(t, map[string]any{"mode": "shadow"})
	if _, ok := rejected().headers()["x-bifrost-cache-hit"]; ok {
		t.Error("shadow mode must not rewrite cache headers")
	}
}

func TestNoSecondRegeneration(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer srv.Close()
	mustInit(t, map[string]any{"loopback_url": srv.URL})
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(`{}`),
		Headers: map[string]string{loopbackMark: "1"}}
	resp := &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{}, Body: []byte(`{"stale":true}`)}
	d := rejected()
	regenerate(ctx, req, resp, d)
	if calls != 0 || resp.StatusCode != http.StatusBadGateway || d.Regenerated != "failed" {
		t.Errorf("re-loop not blocked: calls=%d status=%d", calls, resp.StatusCode)
	}
}

func TestFailClosedAndNoRedirects(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://example.com/steal", http.StatusTemporaryRedirect)
	}))
	defer srv.Close()
	mustInit(t, map[string]any{"loopback_url": srv.URL})
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(`{}`), Headers: map[string]string{}}
	resp := &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{}, Body: []byte(`{"stale":true}`)}
	regenerate(ctx, req, resp, rejected())
	if resp.StatusCode != http.StatusTemporaryRedirect || strings.Contains(string(resp.Body), "stale") {
		t.Errorf("redirect was followed or stale answer kept: %d %s", resp.StatusCode, resp.Body)
	}

	mustInit(t, map[string]any{"loopback_url": "http://127.0.0.1:1"}) // nothing listening
	resp = &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{}, Body: []byte(`{"stale":true}`)}
	regenerate(ctx, req, resp, rejected())
	if resp.StatusCode != http.StatusBadGateway || strings.Contains(string(resp.Body), "stale") {
		t.Errorf("must fail closed, got %d %s", resp.StatusCode, resp.Body)
	}
}
