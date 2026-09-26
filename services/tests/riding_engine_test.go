// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
)

// riding_engine_test.go is the cross-process proof that a facade-riding engine
// serves through the facade it rides. It boots one real nvpair-proxy, enables
// only the Ollama facade (which wires its riding engines), feeds the proxy a
// discovery snapshot naming this node at its own listen port, and points each
// engine's local backend at a loopback httptest stub:
//
//   - POST /v1/chat/completions with a rider-only model reaches the rider's
//     stub and never Ollama's.
//   - POST /v1/messages with a rider-only model is forwarded verbatim to the
//     rider: Anthropic Messages is granted to riding engines, and a server
//     that does not implement it passes its own error through.
//   - POST /api/chat with a facade-only model reaches Ollama's stub.
//   - GET /api/tags and GET /v1/models merge both engines' inventories, the
//     rider's OpenAI-shaped records re-shaped under the caller's envelope.
//   - POST /api/chat with the rider-only model is refused locally with the
//     actionable engine-dialect-mismatch 502 naming the OpenAI paths.
//
// The engines here are httptest servers standing in for Ollama and a
// user-managed OpenAI-compatible server; the riding wiring is the real
// facade/enable plus node/set-local-backend choreography the broker performs.

func TestRidingEngineFacadeCrossProcess(t *testing.T) {
	var ollamaChats, riderCalls, riderMessages atomic.Int32

	ollamaStub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
			_, _ = io.WriteString(w, `{"models":[{"name":"ollama-model","model":"ollama-model"}]}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"ollama-model"}]}`)
		case r.Method == http.MethodPost && r.URL.Path == "/api/chat":
			ollamaChats.Add(1)
			_, _ = io.WriteString(w, `{"model":"ollama-model","done":true}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(ollamaStub.Close)

	// A user-managed OpenAI-compatible server speaks the OpenAI dialect — its
	// model list and inference live under /v1 — and, like modern vLLM builds,
	// answers Anthropic Messages on /v1/messages as well.
	riderStub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
			_, _ = io.WriteString(w, `{"data":[{"id":"vllm-model"}]}`)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			riderCalls.Add(1)
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"from the rider"}}]}`)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
			riderMessages.Add(1)
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"from the rider"}],"stop_reason":"end_turn"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(riderStub.Close)

	p := startProxyProc(t, t.TempDir(), freePort(t))
	defer p.stop()

	// The self record points at this proxy's own listener, the way the broker's
	// discovery relay does for the local node, and attributes each engine's
	// models under its own name.
	p.notify(engines.AddressMethod("ollama", "discovery:nodes"), map[string]any{
		"nodes": []map[string]any{{
			"hostUuid": "self-node",
			"name":     "self-node",
			"ip":       "127.0.0.1",
			"trusted":  true,
			"services": map[string]any{"ol": map[string]any{"port": p.port}},
			"modelsByEngine": map[string]any{
				"ollama":            []string{"ollama-model"},
				"openai-compatible": []string{"vllm-model"},
			},
			"lastSeen": time.Now().Unix(),
		}},
	})

	// Backends are addressed to the facade, the way the broker sends them; the
	// payload keeps the rider's own name.
	p.call(engines.AddressMethod("ollama", "node/set-local-backend"), map[string]any{
		"engine": "ollama", "host": "127.0.0.1",
		"port": portOfURL(t, ollamaStub.URL), "healthy": true,
		"models": []string{"ollama-model"},
	})
	p.call(engines.AddressMethod("ollama", "node/set-local-backend"), map[string]any{
		"engine": "openai-compatible", "host": "127.0.0.1",
		"port": portOfURL(t, riderStub.URL), "healthy": true,
		"models": []string{"vllm-model"},
	})
	p.waitForRoutableNode(t)

	client := &http.Client{Timeout: 5 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	base := fmt.Sprintf("http://127.0.0.1:%d", p.port)

	t.Run("openai inference reaches the riding backend", func(t *testing.T) {
		before := riderCalls.Load()
		resp, err := client.Post(base+"/v1/chat/completions", "application/json",
			bytes.NewBufferString(`{"model":"vllm-model","messages":[]}`))
		if err != nil {
			t.Fatalf("rider-model request failed: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("rider-model response = %d %s, want the riding backend's answer", resp.StatusCode, body)
		}
		if got := riderCalls.Load(); got != before+1 {
			t.Fatalf("rider call count = %d, want %d", got, before+1)
		}
		if got := ollamaChats.Load(); got != 0 {
			t.Fatalf("ollama chat count = %d, want 0 (the rider-only model must not reach the facade engine)", got)
		}
	})

	t.Run("anthropic messages reach the riding backend", func(t *testing.T) {
		before := riderMessages.Load()
		resp, err := client.Post(base+"/v1/messages", "application/json",
			bytes.NewBufferString(`{"model":"vllm-model","max_tokens":1,"messages":[]}`))
		if err != nil {
			t.Fatalf("anthropic-model request failed: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("anthropic-model response = %d %s, want the riding backend's answer", resp.StatusCode, body)
		}
		if got := riderMessages.Load(); got != before+1 {
			t.Fatalf("rider message count = %d, want %d", got, before+1)
		}
		if got := ollamaChats.Load(); got != 0 {
			t.Fatalf("ollama chat count = %d, want 0", got)
		}
	})

	t.Run("native inference reaches the facade engine", func(t *testing.T) {
		before := ollamaChats.Load()
		resp, err := client.Post(base+"/api/chat", "application/json",
			bytes.NewBufferString(`{"model":"ollama-model","messages":[]}`))
		if err != nil {
			t.Fatalf("facade-model request failed: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("facade-model response = %d %s, want the ollama backend's answer", resp.StatusCode, body)
		}
		if got := ollamaChats.Load(); got != before+1 {
			t.Fatalf("ollama chat count = %d, want %d", got, before+1)
		}
	})

	t.Run("model lists merge across riding engines", func(t *testing.T) {
		for _, path := range []string{"/api/tags", "/v1/models"} {
			resp, err := client.Get(base + path)
			if err != nil {
				t.Fatalf("GET %s failed: %v", path, err)
			}
			body, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET %s = %d %s, want merged 200", path, resp.StatusCode, body)
			}
			if !bytes.Contains(body, []byte("ollama-model")) || !bytes.Contains(body, []byte("vllm-model")) {
				t.Fatalf("GET %s body %s, want both engines' models merged", path, body)
			}
		}
		// The native listing re-shapes the rider's OpenAI records under its
		// own dialect, keyed by the verbatim model id.
		resp, err := client.Get(base + "/api/tags")
		if err != nil {
			t.Fatalf("GET /api/tags failed: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if !bytes.Contains(body, []byte(`"name":"vllm-model"`)) {
			t.Fatalf("GET /api/tags body %s, want the rider record re-shaped to a native name record", body)
		}
	})

	t.Run("native path refuses a rider-only model actionably", func(t *testing.T) {
		resp, err := client.Post(base+"/api/chat", "application/json",
			bytes.NewBufferString(`{"model":"vllm-model","messages":[]}`))
		if err != nil {
			t.Fatalf("dialect-mismatch request failed: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusBadGateway {
			t.Fatalf("dialect-mismatch response = %d %s, want actionable local 502", resp.StatusCode, body)
		}
		for _, want := range []string{"engine-dialect-mismatch", "/v1/chat/completions"} {
			if !bytes.Contains(body, []byte(want)) {
				t.Fatalf("dialect-mismatch body %s, want it to name %q", body, want)
			}
		}
		if got := ollamaChats.Load(); got != 1 {
			t.Fatalf("ollama chat count = %d, want 1 (only the facade-model request may reach it)", got)
		}
	})
}
