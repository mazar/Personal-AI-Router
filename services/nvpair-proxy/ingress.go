// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
)

const engineIdentityProbeHeader = "X-NVPAIR-Engine-Identity-Probe"

// localBackend is the explicit loopback engine the cluster mTLS ingress
// forwards to. It is supplied by the broker over node/set-local-backend and is
// deliberately NOT sourced from the discovery overlay: a request that arrived
// over the LAN mTLS ingress can only ever be dumped on this node's own local
// engine, never re-routed to a peer, so the ingress path is strictly terminal
// and cannot recurse or amplify. A facade stores one entry per engine — the
// facade engine itself, plus each facade-riding engine whose models share
// this listener.
type localBackend struct {
	Engine  string `json:"engine"`
	Host    string `json:"host"`
	Port    int    `json:"port"`
	Healthy bool   `json:"healthy"`
	// Models is the engine's last-known served inventory, pushed by the broker
	// alongside the endpoint. It is what picks which local backend serves an
	// OpenAI-dialect inference call without probing the engines; empty routes
	// nothing by model, and the Healthy flag decides whether the backend is
	// usable at all.
	Models []string `json:"models,omitempty"`
}

// currentBackend snapshots the configured local backend for one engine.
func (f *facade) currentBackend(engine string) (localBackend, bool) {
	f.backendMu.RLock()
	defer f.backendMu.RUnlock()
	b, ok := f.backends[engine]
	return b, ok
}

// setLocalBackend records (or, with a zero port / unhealthy flag, effectively
// clears) the local engine the payload names: the facade engine when the
// payload names none, or one of the engines riding this facade.
//
// A non-loopback host is rejected rather than stored. The ingress forwards a
// pin-authenticated peer's request straight here without consulting discovery,
// so an off-box host would turn this node into a relay to an address chosen by
// whoever can reach the control channel. The broker only ever sends 127.0.0.1;
// this is the same defence-in-depth re-validation setLoopbackAlias performs on
// the alias the broker sends it.
func (f *facade) setLocalBackend(b localBackend) error {
	if b.Host != "" && !isLoopbackHost(b.Host) {
		return fmt.Errorf("local backend host %q is not loopback", b.Host)
	}
	name := b.Engine
	if name == "" {
		name = f.profile.Name
	}
	if name != f.profile.Name {
		rider, ok := profileFor(name)
		if !ok || rider.SharedFacade != f.profile.Name {
			return fmt.Errorf("engine %q does not ride the %s facade", b.Engine, f.profile.Name)
		}
	}
	b.Engine = name
	f.backendMu.Lock()
	defer f.backendMu.Unlock()
	if f.backends == nil {
		f.backends = make(map[string]localBackend)
	}
	f.backends[name] = b
	return nil
}

// localBackendTarget returns the loopback URL of the named engine's local
// backend, and false when none is set/healthy (the ingress then answers 503
// rather than forwarding). The host defaults to 127.0.0.1, and setLocalBackend
// refuses to store anything that is not loopback, so this is always a loopback
// target.
func (f *facade) localBackendTarget(engine string) (*url.URL, bool) {
	b, ok := f.currentBackend(engine)
	if !ok || b.Port <= 0 || !b.Healthy {
		return nil, false
	}
	host := b.Host
	if host == "" {
		host = "127.0.0.1"
	}
	return &url.URL{Scheme: "http", Host: net.JoinHostPort(host, strconv.Itoa(b.Port))}, true
}

// handlePlain is the plaintext personality: it accepts requests only from
// loopback and hands them to the full local router (handleHTTP). A non-loopback
// caller — any LAN peer — is refused; peers must use the mTLS ingress. This is
// what closes the former open-relay exposure (the listener still binds all
// interfaces for the TLS personality, but plaintext is loopback-only).
func (f *facade) handlePlain(w http.ResponseWriter, r *http.Request) {
	if !isLoopbackRemote(r.RemoteAddr) {
		slog.Warn("rejected non-loopback plaintext request; cluster peers must use mTLS",
			"remote", r.RemoteAddr, "method", r.Method, "path", r.URL.Path)
		writeIngressError(w, http.StatusForbidden, "loopback-only",
			"plaintext requests are accepted only from loopback; cluster peers must use the mTLS ingress")
		return
	}
	// Engine-manager marks its private identity/action requests so this
	// compatibility facade can never be mistaken for the local Ollama backend.
	if r.Header.Get(engineIdentityProbeHeader) == "1" {
		writeIngressError(w, http.StatusConflict, "proxy-facade",
			"the compatibility facade is not the "+f.profile.DisplayName+" engine")
		return
	}
	f.handleHTTP(w, r)
}

// handleClusterIngress is the LAN mTLS personality: it authenticates the caller
// against this node's cluster pins and, once the peer is a trusted cluster
// member, forwards the request straight to the local loopback engine — exactly
// like the local plaintext path, with no route filtering. The mTLS pin is the
// sole authorization boundary (a trusted peer is treated like a local client),
// so the two personalities stay behaviorally identical toward the engine. It
// never calls resolveCandidates, so a peer request cannot be re-routed onward.
func (f *facade) handleClusterIngress(w http.ResponseWriter, r *http.Request) {
	// Re-derive membership and pins per request so a cluster left, or a peer
	// paired or removed, after startup is reflected immediately without a proxy
	// restart — a removed peer must stop being accepted right away, which is the
	// whole point of the gate.
	f.host.mesh.Refresh()
	peer, ok := f.host.mesh.VerifyClientPin(r)
	if !ok {
		writeIngressError(w, http.StatusForbidden, "cluster-auth",
			"client certificate is not a pinned member of this node's cluster")
		return
	}
	// Classify before choosing a backend. Model lists merge every local engine
	// this facade serves; an inference call's model decides which of them
	// serves it; everything else forwards to the facade engine verbatim,
	// exactly as the single-backend ingress did.
	rt, classified := f.profile.routeFor(r.Method, r.URL.Path)
	engine := f.profile.Name
	if classified && rt.Role.isModelList() {
		f.serveModelList(w, r, rt.Role, f.localCandidates(f.profile.enginesFor(rt.Engines)))
		return
	}
	if classified && rt.Role == roleInferencePOST {
		// Buffering stays confined to classified inference: those bodies are
		// small JSON, and the model field is what picks the engine. Everything
		// else — including large streaming uploads a peer may forward — keeps
		// the unbuffered path the single-backend ingress had.
		bodyBytes, model := bufferBodyAndModel(r)
		engine = f.backendEngineForModel(rt, model)
		if bodyBytes != nil {
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
	}
	target, ok := f.localBackendTarget(engine)
	if !ok {
		writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
			fmt.Sprintf("no local %s engine is available on this node", engine))
		return
	}
	slog.Debug("cluster ingress forwarding to local backend",
		"peer", peer, "method", r.Method, "path", r.URL.Path, "engine", engine, "target", target.Host)
	f.reverseProxyToLocal(w, r, target)
}

// backendEngineForModel picks the local engine an inference route serves a
// model with. A route granted to the facade engine alone is answered by it;
// a multi-engine route matches the model against each local backend's
// inventory under that engine's naming convention, preferring the facade
// engine when both serve the model — the collision rule the merged model list
// and the self candidates use. A model nothing attributes falls back to the
// facade engine, which is what the single-backend ingress did; a backend with
// no pushed inventory therefore never steals a request from it.
func (f *facade) backendEngineForModel(rt route, model string) string {
	if model != "" {
		for _, ep := range f.profile.enginesFor(rt.Engines) {
			b, ok := f.currentBackend(ep.Name)
			if !ok || !b.Healthy || len(b.Models) == 0 {
				continue
			}
			if inventoryAdvertisesModel(ep, b.Models, model) {
				return ep.Name
			}
		}
	}
	return f.profile.Name
}

// reverseProxyToLocal streams the request to the local engine, preserving
// cancellation (the request context is the proxy's root context, so a client
// disconnect or shutdown tears down the upstream call and stops generation).
func (f *facade) reverseProxyToLocal(w http.ResponseWriter, r *http.Request, target *url.URL) {
	f.newLocalReverseProxy(target).ServeHTTP(w, r)
}

func (f *facade) newLocalReverseProxy(target *url.URL) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
		Transport: f.host.plainHTTPTransport(),
		ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
			slog.Warn("cluster ingress upstream error", "target", target.Host, "err", err)
			writeIngressError(ew, http.StatusBadGateway, "backend-error", "local inference backend error")
		},
	}
}

// isLoopbackRemote reports whether an http.Request RemoteAddr (host:port) is a
// loopback address (127.0.0.0/8 or ::1). An unparseable/empty RemoteAddr is not
// loopback, so it fails closed.
func isLoopbackRemote(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// writeIngressError returns the actual failure without granting browser permissions.
func writeIngressError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	body, err := json.Marshal(map[string]string{"error": msg, "code": code})
	if err != nil {
		body = []byte(`{"error":"ingress error"}`)
	}
	_, _ = w.Write(body)
}
