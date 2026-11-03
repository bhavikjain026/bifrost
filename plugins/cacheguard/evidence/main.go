// cacheguard-evidence: companion plugin for Cache Guard's Best-Before.
//
// When Cache Guard regenerates an answer because a cached claim was stale, it attaches
// the fresh source lines to that one regeneration request. This plugin puts them in
// front of the user's question (small models follow evidence in the user turn far
// more reliably than in a separate system message).
//
// Register it with placement "post_builtin" (after semantic_cache): the cache has
// already computed the request's cache key and embedding from the ORIGINAL question,
// so the evidence-grounded answer is cached under the question users actually ask.
//
// It does nothing for any other request.
package main

import (
	"fmt"
	"os"
	"sync/atomic"

	"github.com/maximhq/bifrost/core/schemas"
)

// Must match plugins/cacheguard/freshness.go.
const (
	evidenceTextKey = schemas.BifrostContextKey("cacheguard-evidence-text")
	evidenceDoneKey = schemas.BifrostContextKey("cacheguard-evidence-done")
)

var debug = os.Getenv("CACHEGUARD_DEBUG") == "1"

func Init(config any) error { return nil }

func GetName() string { return "cacheguard-evidence" }

func Cleanup() error { return nil }

func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	text, _ := ctx.Value(evidenceTextKey).(string)
	done, _ := ctx.Value(evidenceDoneKey).(*atomic.Bool)
	if text == "" || done == nil || req == nil || req.ChatRequest == nil {
		return req, nil, nil
	}
	msgs := append([]schemas.ChatMessage(nil), req.ChatRequest.Input...)
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != schemas.ChatMessageRoleUser || m.Content == nil || m.Content.ContentStr == nil {
			continue
		}
		combined := text + "\nQuestion: " + *m.Content.ContentStr
		m.Content = &schemas.ChatMessageContent{ContentStr: &combined}
		msgs[i] = m
		req.ChatRequest.Input = msgs
		done.Store(true)
		if debug {
			fmt.Fprintf(os.Stderr, "[cacheguard-evidence] added %d bytes of evidence to the user message\n", len(text))
		}
		return req, nil, nil
	}
	if debug {
		fmt.Fprintln(os.Stderr, "[cacheguard-evidence] no plain-text user message to attach evidence to")
	}
	return req, nil, nil
}
