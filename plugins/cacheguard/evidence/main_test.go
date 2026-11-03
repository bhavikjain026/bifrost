package main

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestAddsEvidenceOnlyWhenAsked(t *testing.T) {
	q := "What is the repo rate?"
	mk := func() *schemas.BifrostRequest {
		return &schemas.BifrostRequest{RequestType: schemas.ChatCompletionRequest, ChatRequest: &schemas.BifrostChatRequest{
			Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &q}}}}}
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req, _, _ := PreLLMHook(ctx, mk())
	if len(req.ChatRequest.Input) != 1 {
		t.Fatal("must not change ordinary requests")
	}
	done := new(atomic.Bool)
	ctx.SetValue(evidenceTextKey, "[cacheguard-evidence] repo rate is 5.50%")
	ctx.SetValue(evidenceDoneKey, done)
	orig := mk()
	req, _, _ = PreLLMHook(ctx, orig)
	got := *req.ChatRequest.Input[0].Content.ContentStr
	if len(req.ChatRequest.Input) != 1 || !done.Load() || !strings.HasPrefix(got, "[cacheguard-evidence]") || !strings.HasSuffix(got, "Question: "+q) {
		t.Fatalf("evidence not added in front of the question: %q", got)
	}
}
