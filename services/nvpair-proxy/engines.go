// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// This is the only file in the proxy that carries table-driven engine values.
// Everything else reads them from the profile it is handed.
//
// Cross-process identity — Name, DisplayName, DiscoveryService, Namespace,
// FacadePort, EnginePortBase — lives in nvpair-shared/engines, because the
// broker, the scheduler and the TUI need the same values and a second
// declaration here could drift from theirs without failing to compile. What
// remains below is what only the proxy needs to know.
//
// Adding an engine is one entry here plus one in the broker's table. The
// values that are derivable are derived: the applog component and the error-ID
// prefix both come from Name via ComponentName, so they cannot disagree with
// each other. The persisted-port filename is declared instead, because
// Ollama's predates the multi-engine world — see PortFile in
// nvpair-shared/engines.

import (
	"slices"
	"strings"

	"nvpair-shared/engines"
)

// routeRole classifies an inbound request path. It carries the HTTP method
// because the method is part of the classification, not a separate axis: a
// POST to /v1/models is not a model list, and a GET to /api/chat is not
// inference. Folding them into one constant keeps the two from disagreeing.
//
// The dialect distinction is meaningful only for model-list roles. All of
// Ollama's inference paths are handled identically — no envelope, identity
// field or response shape is selected — so there is deliberately no
// per-dialect inference role.
type routeRole int

const (
	// roleInferencePOST is a POST that counts as a cluster workload: it is
	// routed to an owner of the requested model, emits workload lifecycle
	// notifications, and retries a 404 on another owner.
	roleInferencePOST routeRole = iota

	// roleModelListNativeGET is Ollama's /api/tags dialect: the upstream
	// envelope carries a "models" array, each record's identity is its
	// "model" field falling back to "name", and the federated response is
	// re-emitted as {"models": [...]}.
	roleModelListNativeGET

	// roleModelListOpenAIGET is the OpenAI /v1/models dialect: the upstream
	// envelope carries a "data" array, each record's identity is its "id",
	// and the federated response is re-emitted as
	// {"object": "list", "data": [...]}.
	roleModelListOpenAIGET
)

// modelNaming is how an engine spells a model identifier. It governs the
// native model list's dedupe key and nodeAdvertisesModel's match against a
// node's advertised inventory; those two must agree there, or the proxy can
// list a model it then refuses to route.
//
// The OpenAI list path (roleModelListOpenAIGET) is deliberately outside that
// pairing: it dedupes on the upstream identity verbatim, as the pre-unification
// proxies both did, while nodeAdvertisesModel still normalizes. Widening the
// claim to "both list paths" would describe code that does not exist.
type modelNaming int

const (
	// impliedLatestTag is Ollama's convention: an untagged name means the
	// ":latest" tag, so "llama3" and "llama3:latest" are one model.
	impliedLatestTag modelNaming = iota

	// exactID compares identifiers byte for byte.
	exactID
)

// route is one classified request path.
type route struct {
	Path string
	Role routeRole
	// Engines names the engines allowed to serve this path, facade engine
	// first. Nil means the facade engine alone: every route on a facade
	// without riding engines, and on Ollama's facade the native dialect its
	// riders do not speak.
	Engines []string
}

// engineProfile is everything the proxy needs to front one engine.
type engineProfile struct {
	engines.Engine

	// StandalonePort is the port the proxy binds when nothing passes --port
	// and no port has been persisted. It must be neither FacadePort (which
	// the engine itself owns) nor a port an engine is relocated onto.
	//
	// Ollama's 11435 currently equals its EnginePortBase, which violates that
	// second half. It is left as-is because it is the value on disk in every
	// existing install; see the plan's port-field note.
	StandalonePort int

	// Routes classifies the paths this engine's clients call. It is NOT an
	// allowlist: an unlisted path is forwarded verbatim, which is how
	// /api/show, /api/pull, /api/ps, /api/version and OPTIONS preflights keep
	// working.
	Routes []route

	// ModelNaming is how this engine spells model identifiers.
	ModelNaming modelNaming

	// ReservedPersistedPort is a port that must never be restored from the
	// persisted-port file even if it is stored there, because it belongs to
	// the engine rather than the proxy. Zero means no port is reserved.
	ReservedPersistedPort int

	// SupportsHostAlias reports whether facade/enable's aliasAddresses apply to
	// this engine. Only Ollama has an inherited host variable (OLLAMA_HOST) for the
	// alias to stand in for, and the warning the proxy raises when it cannot
	// claim one names that variable. Gating on this makes the scoping
	// enforced rather than left to the broker's restraint in passing the flag.
	SupportsHostAlias bool

	// RidingEngines lists the facade-less engines whose models this facade
	// serves alongside the facade engine's own. Empty for every facade but
	// Ollama's, which carries the user-managed OpenAI-compatible server.
	// Riders never appear in profiles — they front no listener of their own,
	// so enableFacade refuses them — and this list is the only way their
	// profiles are reached.
	RidingEngines []engineProfile
}

// ollamaFacadeEngines is the engine grant on the Ollama facade's shared
// routes: the facade engine first, then the facade-riding OpenAI-compatible
// server. Declared before the route tables because they name it; a test pins
// the names against the shared engine table.
var ollamaFacadeEngines = []string{"ollama", "openai-compatible"}

// ollamaBaseRoutes is the engine-specific surface that Ollama exposes before
// the shared compatibility routes are added. The native inference routes
// carry no engine grant: they are Ollama's dialect alone. The two model lists
// are granted to both engines, because a merged list is what answers them.
var ollamaBaseRoutes = []route{
	{Path: "/api/generate", Role: roleInferencePOST},
	{Path: "/api/chat", Role: roleInferencePOST},
	{Path: "/api/embeddings", Role: roleInferencePOST},
	{Path: "/api/embed", Role: roleInferencePOST},
	{Path: "/api/tags", Role: roleModelListNativeGET, Engines: ollamaFacadeEngines},
	{Path: "/v1/models", Role: roleModelListOpenAIGET, Engines: ollamaFacadeEngines},
}

// lmStudioBaseRoutes is the engine-specific surface that LM Studio exposes
// before the shared compatibility routes are added.
var lmStudioBaseRoutes = []route{
	{Path: "/v1/models", Role: roleModelListOpenAIGET},
}

// openAICompatibleBaseRoutes is the surface a facade-riding OpenAI-compatible
// server exposes beyond the shared inference routes.
var openAICompatibleBaseRoutes = []route{
	{Path: "/v1/models", Role: roleModelListOpenAIGET},
}

// openAIInferenceRoutes is the OpenAI-compatible inference surface.
var openAIInferenceRoutes = []route{
	{Path: "/v1/chat/completions", Role: roleInferencePOST},
	{Path: "/v1/completions", Role: roleInferencePOST},
	{Path: "/v1/embeddings", Role: roleInferencePOST},
}

// anthropicInferenceRoutes is the Anthropic-compatible inference surface.
var anthropicInferenceRoutes = []route{
	{Path: "/v1/messages", Role: roleInferencePOST},
}

var profiles = buildProfiles()

func buildProfiles() []engineProfile {
	ollama, _ := engines.ByName("ollama")
	lmstudio, _ := engines.ByName("lmstudio")
	openAICompatible, ok := engines.ByName("openai-compatible")
	if !ok {
		panic("nvpair-proxy: the shared engine table has no openai-compatible entry")
	}
	// The Ollama facade serves two engines: its own, and the user-managed
	// OpenAI-compatible server riding it. Its OpenAI-dialect inference routes
	// and Anthropic Messages are granted to both: a modern OpenAI-compatible
	// server (vLLM, llama.cpp, SGLang) serves Anthropic Messages as well, and
	// the request is forwarded verbatim, so a rider that does not implement it
	// passes its own error through. The native routes stay Ollama's alone,
	// because the rider speaks none of them.
	ollamaRoutes := slices.Concat(
		ollamaBaseRoutes,
		routesServedBy(openAIInferenceRoutes, ollamaFacadeEngines),
		routesServedBy(anthropicInferenceRoutes, ollamaFacadeEngines),
	)
	lmStudioRoutes := slices.Concat(lmStudioBaseRoutes, openAIInferenceRoutes, anthropicInferenceRoutes)

	return []engineProfile{
		{
			Engine:                ollama,
			StandalonePort:        11435,
			Routes:                ollamaRoutes,
			ModelNaming:           impliedLatestTag,
			ReservedPersistedPort: 0,
			SupportsHostAlias:     true,
			RidingEngines: []engineProfile{
				{
					Engine:      openAICompatible,
					Routes:      slices.Concat(openAICompatibleBaseRoutes, openAIInferenceRoutes),
					ModelNaming: exactID,
				},
			},
		},
		{
			Engine:         lmstudio,
			StandalonePort: 1234,
			Routes:         lmStudioRoutes,
			ModelNaming:    exactID,
			// 1235 is where engine-manager runs a managed LM Studio, so a
			// proxy that restored it would sit on the engine's own port. The
			// stored value predates the current default of 1234.
			ReservedPersistedPort: 1235,
		},
	}
}

// routesServedBy copies a route table with an explicit engine grant attached.
// The un-annotated tables stay grant-free (nil = the facade engine alone), so
// the multi-engine grant is visible at exactly the routes that have one.
func routesServedBy(routes []route, engines []string) []route {
	out := make([]route, len(routes))
	for i, r := range routes {
		out[i] = route{Path: r.Path, Role: r.Role, Engines: engines}
	}
	return out
}

// allProfiles returns every engine the proxy knows: each facade engine
// followed by the engines riding it. Riders front no listener, so they never
// appear in profiles — enableFacade refuses them — but their dialect and
// naming are first-class through here.
func allProfiles() []engineProfile {
	out := make([]engineProfile, 0, len(profiles)+1)
	for _, p := range profiles {
		out = append(out, p)
		out = append(out, p.RidingEngines...)
	}
	return out
}

// profileFor resolves the engine named in a facade/enable request or a
// node/set-local-backend payload: a facade engine, or one riding a facade.
func profileFor(name string) (engineProfile, bool) {
	for _, p := range allProfiles() {
		if p.Name == name {
			return p, true
		}
	}
	return engineProfile{}, false
}

// engineNames lists the accepted engine ids for a facade/enable rejection.
func engineNames() string {
	names := make([]string, 0, len(profiles))
	for _, p := range profiles {
		names = append(names, p.Name)
	}
	return strings.Join(names, ", ")
}

// routeFor classifies a request and returns the matched route, whose Engines
// say who may serve it. The bool reports whether the path is one this engine
// handles specially; false means forward it verbatim.
func (p engineProfile) routeFor(method, path string) (route, bool) {
	for _, r := range p.Routes {
		// Keep scanning on a method mismatch rather than bailing: a path may
		// legitimately appear twice under different methods, and returning
		// here would make the second entry permanently unreachable — silently
		// forwarded verbatim instead of classified.
		if r.Path != path || r.Role.method() != method {
			continue
		}
		return r, true
	}
	return route{}, false
}

// roleFor classifies a request. The bool reports whether the path is one this
// engine handles specially; false means forward it verbatim.
func (p engineProfile) roleFor(method, path string) (routeRole, bool) {
	r, ok := p.routeFor(method, path)
	if !ok {
		return 0, false
	}
	return r.Role, true
}

// enginesFor resolves the engines a route's grant allows on this facade,
// facade engine first. An empty grant means the facade engine alone. A grant
// naming only engines this facade does not host degrades to the facade
// engine rather than resolving to nothing — a table drift then costs a
// mis-routed request, not a silently dead route.
func (p engineProfile) enginesFor(allowed []string) []engineProfile {
	if len(allowed) == 0 {
		return []engineProfile{p}
	}
	out := make([]engineProfile, 0, len(allowed))
	if slices.Contains(allowed, p.Name) {
		out = append(out, p)
	}
	for _, rider := range p.RidingEngines {
		if slices.Contains(allowed, rider.Name) {
			out = append(out, rider)
		}
	}
	if len(out) == 0 {
		return []engineProfile{p}
	}
	return out
}

// method is the HTTP method a role applies to.
func (r routeRole) method() string {
	if r == roleInferencePOST {
		return "POST"
	}
	return "GET"
}

// isModelList reports whether a role serves the federated model list.
func (r routeRole) isModelList() bool {
	return r == roleModelListNativeGET || r == roleModelListOpenAIGET
}

// normalizeModel applies the engine's naming convention to a model
// identifier, so the federated list's dedupe key and the routing gate agree.
func (p engineProfile) normalizeModel(model string) string {
	model = strings.TrimSpace(model)
	if p.ModelNaming != impliedLatestTag || model == "" {
		return model
	}
	name := model[strings.LastIndex(model, "/")+1:]
	if name != "" && !strings.ContainsAny(name, ":@") {
		return model + ":latest"
	}
	return model
}
