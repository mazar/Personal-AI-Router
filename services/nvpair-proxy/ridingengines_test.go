// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for the OpenAI-compatible engine riding the Ollama facade: per-engine
// model attribution, the self-target backend fan-out, the merged model lists,
// and the dialect boundary between the two engines sharing one port.
//
// The engineCase harness deliberately does not carry this engine: it enumerates
// facade-owning engines, and a rider has no facade to front. These tests name
// the rider explicitly, because its whole behavior is defined by riding.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"nvpair-shared/noderec"
)

// riderProfile resolves the facade-riding engine. Every test in this file needs
// it, and a missing entry in the shared table would fail here rather than as a
// zero-value profile silently matching nothing.
func riderProfile(t *testing.T) engineProfile {
	t.Helper()
	p, ok := profileFor("openai-compatible")
	if !ok {
		t.Fatal("openai-compatible profile missing")
	}
	return p
}

// ollamaFacade is a test proxy for the Ollama facade, which is the only facade
// that hosts riding engines today.
func ollamaFacade(t *testing.T, disc *Discovery) *facade {
	t.Helper()
	ollama, ok := profileFor("ollama")
	if !ok {
		t.Fatal("ollama profile missing")
	}
	return testProxy(ollama, disc, ollama.FacadePort).soleFacade()
}

// loopbackRequest is an httptest request from a loopback caller, which is what
// the plaintext personality accepts.
func loopbackRequest(method, path, body string) *http.Request {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:40000"
	return req
}

// A node's per-engine inventory is matched under each engine's own naming
// convention, and a model both engines advertise resolves facade-first — the
// collision rule the self fan-out, the ingress engine pick, and the merged
// model list all share.
func TestEnginesAdvertisingNormalizesPerEngine(t *testing.T) {
	ollama, ok := profileFor("ollama")
	if !ok {
		t.Fatal("ollama profile missing")
	}
	rider := riderProfile(t)
	both := []engineProfile{ollama, rider}

	n := Node{
		ModelsByEngine: map[string][]string{
			"ollama":            {"qwen3", "llama3:8b"},
			"openai-compatible": {"vllm-model", "qwen3-8b"},
		},
	}

	// The facade's implied-latest-tag convention matches the untagged request;
	// the rider's list does not carry it.
	got := enginesAdvertising(both, n, "qwen3")
	if len(got) != 1 || got[0].Name != "ollama" {
		t.Fatalf("qwen3 advertising engines = %v, want [ollama]", got)
	}

	// The rider's ids match byte for byte only.
	got = enginesAdvertising(both, n, "vllm-model")
	if len(got) != 1 || got[0].Name != "openai-compatible" {
		t.Fatalf("vllm-model advertising engines = %v, want [openai-compatible]", got)
	}
	got = enginesAdvertising(both, n, "qwen3-8b")
	if len(got) != 1 || got[0].Name != "openai-compatible" {
		t.Fatalf("qwen3-8b advertising engines = %v, want [openai-compatible]", got)
	}

	// exactID means the rider does not treat an untagged name as its tagged
	// form the way the facade engine does.
	tagged := Node{ModelsByEngine: map[string][]string{"openai-compatible": {"mistral"}}}
	if got := enginesAdvertising(both, tagged, "mistral:latest"); len(got) != 0 {
		t.Fatalf("mistral:latest advertising engines = %v, want none (exactID)", got)
	}

	// A model both engines serve resolves facade first, whatever map iteration
	// order the inventory came from.
	shared := Node{ModelsByEngine: map[string][]string{
		"ollama":            {"shared"},
		"openai-compatible": {"shared"},
	}}
	got = enginesAdvertising(both, shared, "shared")
	if len(got) != 2 || got[0].Name != "ollama" || got[1].Name != "openai-compatible" {
		t.Fatalf("shared advertising engines = %v, want [ollama openai-compatible]", got)
	}
}

// nodeAdvertisesModel's legacy fallback is facade-engine-only: a node that
// sends no per-engine attribution may still serve its flat list as the facade
// engine's, but never as a rider's — a peer without attribution predates
// facade-riding engines and serves none.
func TestNodeAdvertisesModelFallbackIsFacadeOnly(t *testing.T) {
	ollama, ok := profileFor("ollama")
	if !ok {
		t.Fatal("ollama profile missing")
	}
	rider := riderProfile(t)

	legacy := Node{Models: []string{"qwen3"}}
	if !nodeAdvertisesModel(ollama, legacy, "qwen3") {
		t.Error("flat inventory should still advertise for the facade engine")
	}
	if nodeAdvertisesModel(rider, legacy, "qwen3") {
		t.Error("flat inventory must not advertise for a facade-riding engine")
	}

	// Once attribution exists it is authoritative: an engine absent from it
	// serves nothing here, even though the flat list would have matched.
	attributed := Node{
		Models: []string{"qwen3"},
		ModelsByEngine: map[string][]string{
			"openai-compatible": {"vllm-model"},
		},
	}
	if nodeAdvertisesModel(ollama, attributed, "qwen3") {
		t.Error("attribution present, facade absent: the flat list must not fill in")
	}
	if !nodeAdvertisesModel(rider, attributed, "vllm-model") {
		t.Error("attributed rider inventory should advertise")
	}
}

// A self-target node expands into one candidate per healthy local backend the
// route grants — facade engine first — and a model-routed request expands only
// over the engines advertising the model, so the first candidate is the engine
// that will actually serve it.
func TestResolveCandidatesSelfTargetServesPerEngine(t *testing.T) {
	disc := NewDiscovery()
	disc.SetSubscribed([]Node{{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      11434,
		ModelsByEngine: map[string][]string{
			"ollama":            {"qwen3", "llama3:8b"},
			"openai-compatible": {"vllm-model", "qwen3"},
		},
	}})
	f := ollamaFacade(t, disc)
	if err := f.setLocalBackend(localBackend{Engine: "ollama", Host: "127.0.0.1", Port: 11435, Healthy: true}); err != nil {
		t.Fatal(err)
	}
	if err := f.setLocalBackend(localBackend{Engine: "openai-compatible", Host: "127.0.0.1", Port: 8000, Healthy: true}); err != nil {
		t.Fatal(err)
	}

	// A rider-only model on a route granted to both engines yields exactly the
	// rider's backend, so the workload attributes to openai-compatible and no
	// 404 is spent on Ollama's.
	cands := f.resolveCandidates("vllm-model", ollamaFacadeEngines...)
	if len(cands) != 1 {
		t.Fatalf("rider-only candidates = %+v, want just the rider backend", cands)
	}
	if cands[0].engine != "openai-compatible" || cands[0].url.Port() != "8000" || cands[0].id != "self" {
		t.Fatalf("rider-only candidate = %+v, want the openai-compatible backend on :8000", cands[0])
	}

	// A facade-only model yields exactly the facade engine's backend — on the
	// native route's facade-only grant and on the shared grant alike.
	cands = f.resolveCandidates("llama3:8b")
	if len(cands) != 1 || cands[0].engine != "ollama" || cands[0].url.Port() != "11435" {
		t.Fatalf("facade-only candidates = %+v, want just the ollama backend on :11435", cands)
	}
	cands = f.resolveCandidates("llama3:8b", ollamaFacadeEngines...)
	if len(cands) != 1 || cands[0].engine != "ollama" {
		t.Fatalf("facade-only model on a shared grant = %+v, want just ollama", cands)
	}

	// A model both engines serve lists the facade engine first — the collision
	// rule — with the rider as failover.
	cands = f.resolveCandidates("qwen3", ollamaFacadeEngines...)
	if len(cands) != 2 || cands[0].engine != "ollama" || cands[1].engine != "openai-compatible" {
		t.Fatalf("shared-model candidates = %+v, want [ollama openai-compatible]", cands)
	}

	// On a route granted to the facade engine alone, the rider's advertisement
	// is invisible: the shared model resolves to ollama only, and the
	// rider-only model resolves to nothing at all.
	cands = f.resolveCandidates("qwen3")
	if len(cands) != 1 || cands[0].engine != "ollama" {
		t.Fatalf("facade-granted shared model = %+v, want just ollama", cands)
	}
	if cands := f.resolveCandidates("vllm-model", "ollama"); len(cands) != 0 {
		t.Fatalf("facade-granted rider-only model resolved %+v, want nothing", cands)
	}

	// A backend's health flips only its own leg: an unhealthy rider leaves the
	// node attributed but the expansion empty, so routing falls through to the
	// generic no-owner rejection rather than dispatching to a dead port.
	if err := f.setLocalBackend(localBackend{Engine: "openai-compatible", Host: "127.0.0.1", Port: 8000, Healthy: false}); err != nil {
		t.Fatal(err)
	}
	if cands := f.resolveCandidates("vllm-model", ollamaFacadeEngines...); len(cands) != 0 {
		t.Fatalf("unhealthy rider resolved %+v, want nothing", cands)
	}
	if cands := f.resolveCandidates("llama3:8b", ollamaFacadeEngines...); len(cands) != 1 || cands[0].engine != "ollama" {
		t.Fatalf("unhealthy rider disturbed the facade engine's candidates: %+v", cands)
	}
}

// An Ollama-native inference path cannot reach a model only the riding engine
// serves. The merged model list shows the model, so the rejection must say the
// model exists and name the path that reaches it, rather than claiming no node
// advertises it.
func TestHandlePlainNativePathRejectsRiderOnlyModelActionably(t *testing.T) {
	disc := NewDiscovery()
	disc.SetSubscribed([]Node{{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      11434,
		ModelsByEngine: map[string][]string{
			"openai-compatible": {"vllm-model"},
		},
	}})
	p := testProxy(mustProfile(t, "ollama"), disc, 11434)
	f := p.soleFacade()

	rec := httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodPost, "/api/chat", `{"model":"vllm-model"}`))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("native-path rider model status = %d, want %d", rec.Code, http.StatusBadGateway)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "engine-dialect-mismatch") {
		t.Errorf("rejection body %q does not carry the engine-dialect-mismatch code", body)
	}
	if !strings.Contains(body, "/v1/chat/completions") {
		t.Errorf("rejection body %q does not name the path that reaches the model", body)
	}
	if !strings.Contains(body, "vllm-model") {
		t.Errorf("rejection body %q does not name the model", body)
	}

	// A facade-only model with no reachable backend keeps the generic
	// no-owner rejection: the model is known to this node, its engine is just
	// down — nothing about dialects is wrong.
	disc2 := NewDiscovery()
	disc2.SetSubscribed([]Node{{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      11434,
		ModelsByEngine: map[string][]string{
			"ollama": {"qwen3"},
		},
	}})
	f2 := testProxy(mustProfile(t, "ollama"), disc2, 11434).soleFacade()
	rec2 := httptest.NewRecorder()
	f2.handlePlain(rec2, loopbackRequest(http.MethodPost, "/api/chat", `{"model":"qwen3"}`))
	if rec2.Code != http.StatusBadGateway {
		t.Fatalf("unavailable facade model status = %d, want %d", rec2.Code, http.StatusBadGateway)
	}
	if strings.Contains(rec2.Body.String(), "engine-dialect-mismatch") {
		t.Errorf("a facade-only model must not be reported as a dialect mismatch: %q", rec2.Body.String())
	}
	if !strings.Contains(rec2.Body.String(), "no available node advertises the requested model") {
		t.Errorf("rejection body %q lost the no-owner message", rec2.Body.String())
	}

	// A route that already reaches the rider never reports a dialect mismatch:
	// with the grant satisfied, an empty resolution is an availability
	// problem.
	rec3 := httptest.NewRecorder()
	f.handlePlain(rec3, loopbackRequest(http.MethodPost, "/v1/chat/completions", `{"model":"vllm-model"}`))
	if rec3.Code != http.StatusBadGateway {
		t.Fatalf("granted-rider-without-backend status = %d, want %d", rec3.Code, http.StatusBadGateway)
	}
	if strings.Contains(rec3.Body.String(), "engine-dialect-mismatch") {
		t.Errorf("a granted rider must not be reported as a dialect mismatch: %q", rec3.Body.String())
	}
}

// mustProfile resolves an engine profile, failing with a message instead of
// silently testing a zero value.
func mustProfile(t *testing.T, name string) engineProfile {
	t.Helper()
	p, ok := profileFor(name)
	if !ok {
		t.Fatalf("%s profile missing", name)
	}
	return p
}

// OpenAI-dialect inference on the shared facade port reaches the riding
// engine's loopback server when that engine advertises the model — and the
// native path still refuses it even while the backend is healthy, because the
// dialect, not availability, is what gates the native path.
func TestHandlePlainOpenAIInferenceReachesRidingBackend(t *testing.T) {
	var riderHits atomic.Int32
	rider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("rider received %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		riderHits.Add(1)
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("rider request body: %v", err)
		}
		if body.Model != "vllm-model" {
			t.Errorf("rider received model %q, want vllm-model", body.Model)
		}
		w.Write([]byte(`{"choices":[]}`))
	}))
	defer rider.Close()

	disc := NewDiscovery()
	disc.SetSubscribed([]Node{{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      11434,
		ModelsByEngine: map[string][]string{
			"openai-compatible": {"vllm-model"},
		},
	}})
	f := ollamaFacade(t, disc)
	riderBackend := nodeFor(t, "rider-backend", rider.URL)
	if err := f.setLocalBackend(localBackend{
		Engine:  "openai-compatible",
		Host:    riderBackend.Addresses[0],
		Port:    riderBackend.Port,
		Healthy: true, Models: []string{"vllm-model"},
	}); err != nil {
		t.Fatal(err)
	}

	rec := httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodPost, "/v1/chat/completions", `{"model":"vllm-model"}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("rider inference status = %d, want %d: %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if got := riderHits.Load(); got != 1 {
		t.Fatalf("rider received %d inference calls, want 1", got)
	}

	// The native path refuses the same model while the backend is healthy.
	rec = httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodPost, "/api/chat", `{"model":"vllm-model"}`))
	if rec.Code != http.StatusBadGateway || !strings.Contains(rec.Body.String(), "engine-dialect-mismatch") {
		t.Fatalf("native path with healthy rider = %d %q, want the dialect-mismatch refusal", rec.Code, rec.Body.String())
	}
	if got := riderHits.Load(); got != 1 {
		t.Fatalf("refused native path reached the rider %d times, want 0 more", got-1)
	}
}

// GET /api/tags and GET /v1/models on the shared facade port merge both local
// engines' inventories. A rider's records are fetched in its own OpenAI dialect
// and re-shaped when the caller speaks the native one, and a model both engines
// serve keeps the facade engine's record.
func TestHandlePlainMergesModelListsAcrossRidingEngines(t *testing.T) {
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			w.Write([]byte(`{"models":[{"name":"qwen3","model":"qwen3:latest","digest":"aaa"}]}`))
		case "/v1/models":
			w.Write([]byte(`{"data":[{"id":"qwen3"}]}`))
		default:
			t.Errorf("ollama backend received unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer ollamaSrv.Close()

	var riderHits atomic.Int32
	riderSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			// A native-dialect fetch landing here means the fan-out served the
			// rider a path it does not speak.
			t.Errorf("rider backend received unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		riderHits.Add(1)
		w.Write([]byte(`{"data":[{"id":"vllm-model"},{"id":"qwen3"}]}`))
	}))
	defer riderSrv.Close()

	disc := NewDiscovery()
	disc.SetSubscribed([]Node{{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      11434,
		ModelsByEngine: map[string][]string{
			"ollama":            {"qwen3"},
			"openai-compatible": {"vllm-model", "qwen3"},
		},
	}})
	f := ollamaFacade(t, disc)
	ollamaBackend := nodeFor(t, "ollama-backend", ollamaSrv.URL)
	riderBackend := nodeFor(t, "rider-backend", riderSrv.URL)
	if err := f.setLocalBackend(localBackend{
		Engine: "ollama", Host: ollamaBackend.Addresses[0], Port: ollamaBackend.Port, Healthy: true,
	}); err != nil {
		t.Fatal(err)
	}
	if err := f.setLocalBackend(localBackend{
		Engine: "openai-compatible", Host: riderBackend.Addresses[0], Port: riderBackend.Port, Healthy: true,
	}); err != nil {
		t.Fatal(err)
	}

	// Native dialect: both engines' models, the rider's record re-shaped to
	// the native identity field, and the shared model present once — the
	// facade engine's record, digest and all.
	rec := httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodGet, "/api/tags", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("/api/tags status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var native struct {
		Models []struct {
			Name   string `json:"name"`
			Digest string `json:"digest"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &native); err != nil {
		t.Fatalf("/api/tags body: %v", err)
	}
	if len(native.Models) != 2 {
		t.Fatalf("/api/tags merged %d records, want 2: %s", len(native.Models), rec.Body.String())
	}
	if native.Models[0].Name != "qwen3" || native.Models[0].Digest != "aaa" {
		t.Errorf("/api/tags first record = %+v, want the facade engine's qwen3 with its digest", native.Models[0])
	}
	if native.Models[1].Name != "vllm-model" {
		t.Errorf("/api/tags second record = %+v, want the re-shaped vllm-model", native.Models[1])
	}

	// OpenAI dialect: the same merge, both records carried verbatim.
	rec = httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodGet, "/v1/models", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/models status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var openAI struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &openAI); err != nil {
		t.Fatalf("/v1/models body: %v", err)
	}
	if len(openAI.Data) != 2 || openAI.Data[0].ID != "qwen3" || openAI.Data[1].ID != "vllm-model" {
		t.Fatalf("/v1/models merged = %s, want [qwen3 vllm-model]", rec.Body.String())
	}
	if got := riderHits.Load(); got != 2 {
		t.Fatalf("rider served %d model lists, want 2", got)
	}

	// A down rider removes only its own models; the facade engine's list is
	// untouched and the request still succeeds.
	if err := f.setLocalBackend(localBackend{
		Engine: "openai-compatible", Host: riderBackend.Addresses[0], Port: riderBackend.Port, Healthy: false,
	}); err != nil {
		t.Fatal(err)
	}
	before := riderHits.Load()
	rec = httptest.NewRecorder()
	f.handlePlain(rec, loopbackRequest(http.MethodGet, "/api/tags", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("/api/tags with rider down status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	native.Models = nil
	if err := json.Unmarshal(rec.Body.Bytes(), &native); err != nil {
		t.Fatalf("/api/tags body: %v", err)
	}
	if len(native.Models) != 1 || native.Models[0].Name != "qwen3" {
		t.Fatalf("/api/tags with rider down = %s, want only the facade engine's qwen3", rec.Body.String())
	}
	if riderHits.Load() != before {
		t.Error("the down rider's backend was still fetched")
	}
}

// node/set-local-backend stores a facade-riding engine's backend alongside the
// facade engine's, refuses engines that neither own nor ride the facade, and
// keeps the addressing contract: a payload for the rider is addressed to the
// facade engine, and a message addressed to the rider itself finds no facade.
func TestNodeSetLocalBackendAcceptsRidingEngine(t *testing.T) {
	f := ollamaFacade(t, NewDiscovery())

	if err := f.setLocalBackend(localBackend{
		Engine: "openai-compatible", Host: "127.0.0.1", Port: 8000,
		Healthy: true, Models: []string{"vllm-model"},
	}); err != nil {
		t.Fatalf("rider backend refused: %v", err)
	}
	b, ok := f.currentBackend("openai-compatible")
	if !ok || b.Port != 8000 || !b.Healthy || !reflect.DeepEqual(b.Models, []string{"vllm-model"}) {
		t.Fatalf("stored rider backend = %+v ok=%v, want the payload as sent", b, ok)
	}
	if _, ok := f.currentBackend("ollama"); ok {
		t.Error("accepting the rider stored a facade-engine backend too")
	}

	// An engine that neither owns nor rides this facade is refused.
	if err := f.setLocalBackend(localBackend{Engine: "lmstudio", Host: "127.0.0.1", Port: 1234, Healthy: true}); err == nil {
		t.Fatal("lmstudio backend accepted on the ollama facade")
	}

	// A facade with no riding engines refuses everything but itself.
	lf := testProxy(mustProfile(t, "lmstudio"), NewDiscovery(), 1234).soleFacade()
	if err := lf.setLocalBackend(localBackend{Engine: "openai-compatible", Host: "127.0.0.1", Port: 8000, Healthy: true}); err == nil {
		t.Fatal("the lmstudio facade accepted an engine it does not host")
	}
	if err := lf.setLocalBackend(localBackend{Engine: "lmstudio", Host: "127.0.0.1", Port: 1235, Healthy: true}); err != nil {
		t.Fatalf("own backend refused on the lmstudio facade: %v", err)
	}

	// Through the real handler: the broker's payload for the rider is
	// addressed to the facade engine and accepted; addressed to the rider it
	// is refused, because the rider fronts no facade at all.
	rec := &recordingWriter{}
	p2 := newTestProxy(mustProfile(t, "ollama"), NewCodec(rec), NewDiscovery(), 11434)
	id := json.RawMessage(`7`)
	payload := json.RawMessage(`{"engine":"openai-compatible","host":"127.0.0.1","port":8000,"healthy":true}`)
	p2.handleMessage(&Message{
		Method: "ollama:node/set-local-backend",
		Params: payload,
		ID:     &id,
	})
	resp := decodeOneResponse(t, rec)
	if resp.Error != nil {
		t.Fatalf("facade-addressed rider payload rejected: %d %s", resp.Error.Code, resp.Error.Message)
	}

	rec.buf.Reset()
	p2.handleMessage(&Message{
		Method: "openai-compatible:node/set-local-backend",
		Params: payload,
		ID:     &id,
	})
	resp = decodeOneResponse(t, rec)
	if resp.Error == nil || resp.Error.Code != -32002 {
		t.Fatalf("rider-addressed payload answered %+v, want a -32002 facade error", resp.Error)
	}
}

// rpcError is a decoded JSON-RPC error object.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// rpcResponse is one decoded JSON-RPC reply line.
type rpcResponse struct {
	Error *rpcError `json:"error"`
}

// decodeOneResponse parses exactly one reply line from the recorder.
func decodeOneResponse(t *testing.T, rec *recordingWriter) rpcResponse {
	t.Helper()
	lines := rec.lines()
	if len(lines) == 0 {
		t.Fatal("no response line was written")
	}
	if len(lines) != 1 {
		t.Fatalf("%d response lines written, want 1", len(lines))
	}
	var resp rpcResponse
	if err := json.Unmarshal(lines[len(lines)-1], &resp); err != nil {
		t.Fatalf("response line %s: %v", lines[len(lines)-1], err)
	}
	return resp
}

// facade/enable refuses a facade-riding engine with the way out named, and
// enabling the facade engine wires the rider's profile through it.
func TestEnableFacadeRefusesFacadelessEngine(t *testing.T) {
	redirectConfigDir(t)
	rider := riderProfile(t)

	p := NewProxy(NewCodec(rwNop{}))
	p.serveCtx = t.Context()
	t.Cleanup(func() { p.shutdown(t.Context()) })

	if _, err := p.enableFacade(enableFacadeParams{
		Engine:              rider.Name,
		Port:                freeTCPPort(t),
		IgnorePersistedPort: true,
	}); err == nil {
		t.Fatal("enable accepted a facade-riding engine")
	} else if !strings.Contains(err.Error(), `enable "ollama" instead`) {
		t.Errorf("refusal %q does not name the facade that serves the rider", err)
	}
	if f := p.facadeFor(rider.Name); f != nil {
		t.Fatalf("a refused enable left a %s facade running", rider.Name)
	}
	if got := len(p.enabledFacades()); got != 0 {
		t.Fatalf("enabled facades = %d, want none", got)
	}

	// The refusal is about listeners, not identity: the facade engine comes up
	// normally and carries the rider's profile with it.
	port := freeTCPPort(t)
	if _, err := p.enableFacade(enableFacadeParams{
		Engine:              "ollama",
		Port:                port,
		IgnorePersistedPort: true,
	}); err != nil {
		t.Fatalf("enable ollama: %v", err)
	}
	f := p.facadeFor("ollama")
	if f == nil {
		t.Fatal("ollama facade missing after enable")
	}
	if len(f.profile.RidingEngines) != 1 || f.profile.RidingEngines[0].Name != rider.Name {
		t.Fatalf("ollama facade riding engines = %v, want [openai-compatible]", f.profile.RidingEngines)
	}
}

// subscribedToNode keeps the per-engine attribution the routing match reads:
// the facade engine's list from its service record, the rider's from its own
// attribution — and the rider gets no legacy fallback.
func TestSubscribedToNodeProjectsRidingEngineInventory(t *testing.T) {
	ollama := mustProfile(t, "ollama")
	rider := riderProfile(t)

	attributed := noderec.DirectoryNode{
		HostUUID: "uuid-x",
		Name:     "host-x",
		IP:       "10.0.0.9",
		Models:   []string{"stale-union"},
		ModelsByEngine: map[string][]string{
			"ollama":            {"qwen3"},
			"openai-compatible": {"vllm-model"},
		},
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			ollama.DiscoveryService: {Port: ollama.FacadePort},
		},
	}
	got, ok := subscribedToNode(ollama, attributed)
	if !ok {
		t.Fatal("dual-engine node should project")
	}
	want := map[string][]string{
		"ollama":            {"qwen3"},
		"openai-compatible": {"vllm-model"},
	}
	if !reflect.DeepEqual(got.ModelsByEngine, want) {
		t.Fatalf("projected inventory = %v, want %v", got.ModelsByEngine, want)
	}
	// Models keeps the facade engine's list only — the display union a
	// consumer renders is not the routing truth.
	if !reflect.DeepEqual(got.Models, []string{"qwen3"}) {
		t.Fatalf("projected Models = %v, want the facade engine's own [qwen3]", got.Models)
	}
	// And the projection is what routes: the rider's model is a rider-owned
	// candidate, the facade engine does not claim it.
	if !nodeAdvertisesModel(rider, got, "vllm-model") {
		t.Error("projected rider inventory does not advertise its model")
	}
	if nodeAdvertisesModel(ollama, got, "vllm-model") {
		t.Error("facade engine claims a model attributed only to the rider")
	}

	// A peer sending no attribution predates facade-riding engines: its flat
	// list stays the facade engine's, and the rider serves nothing.
	legacy := attributed
	legacy.ModelsByEngine = nil
	legacy.Models = []string{"flat-a", "flat-b"}
	got, ok = subscribedToNode(ollama, legacy)
	if !ok {
		t.Fatal("legacy node should project")
	}
	if !reflect.DeepEqual(got.ModelsByEngine["ollama"], []string{"flat-a", "flat-b"}) {
		t.Fatalf("legacy facade inventory = %v, want the flat list", got.ModelsByEngine["ollama"])
	}
	if len(got.ModelsByEngine["openai-compatible"]) != 0 {
		t.Fatalf("legacy rider inventory = %v, want none", got.ModelsByEngine["openai-compatible"])
	}
	if nodeAdvertisesModel(rider, got, "flat-a") {
		t.Error("a legacy flat list advertised for the rider")
	}
}
