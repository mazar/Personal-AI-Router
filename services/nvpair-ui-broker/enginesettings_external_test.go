// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
)

// externalSettingsFixture stubs the worker for a user-managed engine and the
// proxy for the facade it rides, and answers settings calls the way the real
// components do for an external engine: the launch is never editable, preview
// passes the request through, and configure persists the port and re-probes.
type externalSettingsFixture struct {
	b         *Broker
	configure chan settings.Configure
}

func newExternalSettingsFixture(t *testing.T, facadePort int) *externalSettingsFixture {
	t.Helper()
	f := &externalSettingsFixture{b: &Broker{
		codec:      NewCodec(&bytes.Buffer{}),
		nodeID:     "target",
		clusterDir: filepath.Join(t.TempDir(), "cluster"),
	}, configure: make(chan settings.Configure, 1)}

	worker, workerCodec := newTestRPCWorkerPipe(t)
	f.b.setEngineMgr(worker)
	proxyWorker, proxyCodec := newTestRPCWorkerPipe(t)
	f.b.setProxy(&proxyProcess{peer: proxyWorker.peer, facadeState: map[string]proxyFacadeState{
		"ollama": {ready: true, port: facadePort},
	}})

	externalLaunch := func(port int) settings.LaunchState {
		return settings.LaunchState{
			Engine: "openai-compatible", ServerPort: port, EffectivePort: port,
			Running: true, Adopted: true, External: true,
			Reason: "This engine is user-managed: run and configure it in its own application. PAIR only routes to the server port.",
			Format: "pair-arguments-v1",
		}
	}
	// What the user's server port is right now, as a real engine-manager would
	// report it: the manifest default until a configure persists another one.
	persistedPort := 8000
	go func() {
		codec := workerCodec
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if !msg.IsRequest() {
				continue
			}
			switch msg.Method {
			case "engine:get-launch":
				_ = codec.Respond(msg.ID, externalLaunch(persistedPort))
			case "engine:configured-ports":
				_ = codec.Respond(msg.ID, map[string]any{"engines": []any{}})
			case "engine:preview-launch":
				var p settings.Request
				_ = json.Unmarshal(msg.Params, &p)
				// The worker's external preview is a passthrough: it validates
				// the port and never produces launch text or a restart.
				p.Settings.LaunchText = ""
				_ = codec.Respond(msg.ID, settings.Preview{Settings: p.Settings})
			case "engine:configure-launch":
				var configure settings.Configure
				if json.Unmarshal(msg.Params, &configure) != nil {
					continue
				}
				persistedPort = configure.Settings.ServerPort
				select {
				case f.configure <- configure:
				default:
				}
				_ = codec.Respond(msg.ID, externalLaunch(configure.Settings.ServerPort))
			default:
				_ = codec.RespondError(msg.ID, -32601, "unhandled "+msg.Method)
			}
		}
	}()
	go func() {
		codec := proxyCodec
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if !msg.IsRequest() {
				continue
			}
			_ = codec.Respond(msg.ID, map[string]any{"ok": true})
		}
	}()
	return f
}

// A user-managed engine's settings surface: the snapshot reports the facade it
// rides as its proxy port and stays read-only, a port-only apply persists the
// server port without touching the facade, and a proxy-port change is refused
// because the shared facade's port is not the rider's to configure.
func TestUserManagedEngineSettingsArePortOnly(t *testing.T) {
	const facadePort = 23456
	f := newExternalSettingsFixture(t, facadePort)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	f.b.engineConfigMu.Lock()
	snapshot, err := f.b.settingsSnapshotLocked(ctx, "openai-compatible")
	f.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.External || snapshot.Editable {
		t.Fatalf("external snapshot = %+v, want external and not editable", snapshot)
	}
	if snapshot.Settings.LaunchText != "" {
		t.Fatalf("external snapshot carries launch text %q", snapshot.Settings.LaunchText)
	}
	if snapshot.Settings.ProxyPort != facadePort || snapshot.EffectiveProxyPort != facadePort {
		t.Fatalf("proxy port = %d/%d, want the facade's %d",
			snapshot.Settings.ProxyPort, snapshot.EffectiveProxyPort, facadePort)
	}
	if snapshot.Settings.ServerPort != 8000 {
		t.Fatalf("server port = %d, want the manifest default 8000", snapshot.Settings.ServerPort)
	}
	// A second read reconciles to the same configuration: no revision churn.
	f.b.engineConfigMu.Lock()
	again, err := f.b.settingsSnapshotLocked(ctx, "openai-compatible")
	f.b.engineConfigMu.Unlock()
	if err != nil || again.Revision != snapshot.Revision {
		t.Fatalf("snapshot read churned the revision: %d then %d (%v)", snapshot.Revision, again.Revision, err)
	}

	// Free the server port the apply will use.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	newPort := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	request := settings.Request{Engine: "openai-compatible", ExpectedRevision: snapshot.Revision,
		RequestID: settingsID(), Settings: snapshot.Settings}
	request.Settings.ServerPort = newPort
	preview, err := f.b.previewEngineSettings(ctx, request, "")
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("port-only preview rejected: %+v %v", preview, err)
	}
	if preview.Rebind {
		t.Fatal("a port-only apply must not rebind a facade")
	}

	// The shared facade's port is not the rider's to configure.
	moved := request
	moved.RequestID = settingsID()
	moved.Settings.ProxyPort = 30000
	preview, err = f.b.previewEngineSettings(ctx, moved, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := preview.Errors["ports"]; !ok {
		t.Fatalf("proxy-port change accepted: %+v", preview)
	}

	request.ExpectedRevision = preview.Revision
	receipt, err := f.b.applyEngineSettings(ctx, request, "")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Phase != "succeeded" {
		t.Fatalf("apply receipt = %+v", receipt)
	}
	select {
	case configure := <-f.configure:
		if configure.Settings.ServerPort != newPort || configure.Settings.ProxyPort != facadePort {
			t.Fatalf("configure payload = %+v", configure.Settings)
		}
		if configure.Settings.LaunchText != "" {
			t.Fatal("configure carried launch text for a user-managed engine")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the apply never reached the worker")
	}
	f.b.engineConfigMu.Lock()
	applied, err := f.b.settingsSnapshotLocked(ctx, "openai-compatible")
	f.b.engineConfigMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if applied.Phase != "succeeded" || applied.Settings.ServerPort != newPort {
		t.Fatalf("applied snapshot = %+v", applied)
	}
	if applied.Settings.ProxyPort != facadePort {
		t.Fatalf("apply disturbed the shared facade port: %d", applied.Settings.ProxyPort)
	}
}
