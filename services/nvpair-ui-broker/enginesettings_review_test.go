// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
)

// Adding an engine to the shared table forces a settings decision: either it
// belongs in settingsEngines, or the exclusion is a reviewed conclusion.
func TestSettingsEnginesMatchTheEngineTable(t *testing.T) {
	want := engines.Names()
	slices.Sort(want)
	got := slices.Clone(settingsEngines)
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("settingsEngines = %v, shared engine table = %v; review the new engine's settings story", got, want)
	}
}

// An explicit rider record must be restored onto the rider's own runtime, and
// must never reach the facade-owning engines' state.
func TestPrepareExplicitEngineSettingsSeedsTheRidersOwnRuntime(t *testing.T) {
	b := &Broker{clusterDir: filepath.Join(t.TempDir(), "cluster")}
	b.engineSettingsLoaded = true
	b.engineSettings = map[string]*engineSettingsRecord{
		"openai-compatible": {
			Explicit: true,
			Snapshot: settings.Snapshot{Settings: settings.Config{ServerPort: 9001, ProxyPort: 9002}},
		},
	}
	// Sentinels: if the rider's restore reached the ollama or lmstudio branch,
	// one of these would move.
	b.ollamaState().backendPort.Store(1111)
	b.ollamaState().startupPort.Store(1112)
	b.lmstudioState().backendPort.Store(1113)
	b.lmstudioState().startupPort.Store(1114)

	if !b.prepareExplicitEngineSettings("openai-compatible") {
		t.Fatal("explicit settings were not restored")
	}
	if got := b.engineProxy(openAICompatibleProxyProfile).backendPort.Load(); got != 9001 {
		t.Fatalf("rider backend port = %d, want 9001", got)
	}
	if got := b.engineProxy(openAICompatibleProxyProfile).startupPort.Load(); got != 9002 {
		t.Fatalf("rider startup port = %d, want 9002", got)
	}
	if got := b.ollamaState().backendPort.Load(); got != 1111 {
		t.Fatalf("rider restore moved ollama's backend port to %d", got)
	}
	if got := b.lmstudioState().startupPort.Load(); got != 1114 {
		t.Fatalf("rider restore moved lmstudio's startup port to %d", got)
	}
}

func TestSettingsRebindAddressesOnlyRequestedFacade(t *testing.T) {
	for _, profile := range engineProxyProfiles {
		t.Run(profile.Name, func(t *testing.T) {
			h := newSettingsHarness(t)
			p := h.b.getProxy()
			_, ollamaBefore := p.Status("ollama")
			_, lmstudioBefore := p.Status("lmstudio")
			if profile.SharedFacade != "" {
				// A facade-riding engine has no listener of its own to rebind:
				// its proxy port is the shared facade's, so a rebind must be
				// refused rather than silently applied to the wrong engine.
				if err := h.b.rebindSettingsProxy(profile.Name, 30000); err == nil {
					t.Fatal("a facade-riding engine's rebind was accepted")
				}
				return
			}
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			port := ln.Addr().(*net.TCPAddr).Port
			_ = ln.Close()
			if err := h.b.rebindSettingsProxy(profile.Name, port); err != nil {
				t.Fatal(err)
			}
			wantOllama, wantLMStudio := ollamaBefore, lmstudioBefore
			if profile.Name == "ollama" {
				wantOllama = port
			} else {
				wantLMStudio = port
			}
			for engine, want := range map[string]int{"ollama": wantOllama, "lmstudio": wantLMStudio} {
				ready, got := p.Status(engine)
				if !ready || got != want {
					t.Fatalf("%s ready=%v port=%d, want %d", engine, ready, got, want)
				}
			}
		})
	}
}

func TestExplicitSettingsBindFailurePreservesChosenPort(t *testing.T) {
	for _, profile := range engineProxyProfiles {
		t.Run(profile.Name, func(t *testing.T) {
			if profile.SharedFacade != "" {
				// A facade-riding engine never receives a facade/enable, so
				// there is no bind-failure choreography to protect; its
				// explicit-settings restore is covered by the seeding test.
				t.Skip("no facade of its own to bind")
			}
			const requested = 25000
			b := &Broker{codec: NewCodec(&bytes.Buffer{}), engineSettingsLoaded: true,
				engineSettings: map[string]*engineSettingsRecord{profile.Name: {
					Explicit: true, Snapshot: settings.Snapshot{Settings: settings.Config{ServerPort: 24999, ProxyPort: requested}},
				}},
			}
			if !b.prepareExplicitEngineSettings(profile.Name) {
				t.Fatal("explicit settings were not restored")
			}
			failure := settingsJSON(map[string]any{"code": "bind-failed", "port": requested})
			if profile.Name == "ollama" {
				b.forwardProxyNotification("error", failure)
			} else {
				b.forwardLMStudioProxyNotification("error", failure)
			}
			if got := b.engineProxy(profile).startupPort.Load(); got != requested {
				t.Fatalf("bind notification changed chosen port to %d", got)
			}
			client, server := net.Pipe()
			t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
			p := &proxyProcess{peer: NewPeer(NewCodec(client))}
			go p.peer.Serve(nil, nil)
			attempts := make(chan enableFacadeRequest, 2)
			serveFacadeEnable(t, server, map[int]bool{requested: true}, attempts)
			err := b.enableProxyFacadeWithFallback(context.Background(), p, enableFacadeRequest{Engine: profile.Name, Port: requested}, func(int) int {
				t.Error("explicit port must not fall back")
				return requested + 1
			})
			if err == nil {
				t.Fatal("bind failure was hidden")
			}
			if len(attempts) != 1 {
				t.Fatalf("attempts=%d, want only the chosen port", len(attempts))
			}
		})
	}
}

func TestSettingsReservesStoppedProxySavedPort(t *testing.T) {
	test := func(name string, serverPort bool) {
		t.Run(name, func(t *testing.T) {
			h := newSettingsHarness(t)
			request := h.request(t)
			_, reserved := h.b.getLMStudioProxy().Status("lmstudio")
			h.b.setLMStudioProxy(nil)
			h.b.engineSettings["lmstudio"] = &engineSettingsRecord{
				Explicit: true,
				Snapshot: settings.Snapshot{Settings: settings.Config{ProxyPort: reserved}},
			}
			if serverPort {
				request.Settings.ServerPort = reserved
			} else {
				request.Settings.ProxyPort = reserved
			}
			preview, err := h.b.previewEngineSettings(context.Background(), request, "")
			if err != nil {
				t.Fatal(err)
			}
			if len(preview.Errors) == 0 {
				t.Fatalf("accepted stopped proxy's saved port: %+v", preview)
			}
		})
	}
	test("server port cannot reuse saved proxy port", true)
	test("proxy port cannot reuse saved proxy port", false)
}

func TestSettingsMigrationRejectsReservedPorts(t *testing.T) {
	test := func(name string, reserve func(*settingsHarness, settings.Request) int) {
		t.Run(name, func(t *testing.T) {
			h := newSettingsHarness(t)
			request := h.request(t)
			delete(h.b.engineSettings, "ollama")
			port := reserve(h, request)
			path, err := h.b.engineSettingsPath()
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(map[string]int{"port": port})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(filepath.Dir(path), "proxy-port.json"), data, 0600); err != nil {
				t.Fatal(err)
			}
			h.b.migrateLegacyEngineSettings()
			if _, explicit := h.b.explicitEngineSettings("ollama"); explicit {
				t.Fatal("reserved legacy port became an explicit setting")
			}
		})
	}
	test("PAIR control port", func(h *settingsHarness, request settings.Request) int {
		return engineControlPort
	})
	test("inherited Ollama host alias", func(h *settingsHarness, request settings.Request) int {
		h.b.ollamaHostAliasMu.Lock()
		h.b.ollamaHostAlias.Port = request.Settings.ProxyPort
		h.b.ollamaHostAliasMu.Unlock()
		return request.Settings.ProxyPort
	})
	test("stopped proxy saved port", func(h *settingsHarness, request settings.Request) int {
		_, port := h.b.getLMStudioProxy().Status("lmstudio")
		h.b.setLMStudioProxy(nil)
		h.b.engineSettings["lmstudio"] = &engineSettingsRecord{
			Explicit: true,
			Snapshot: settings.Snapshot{Settings: settings.Config{ProxyPort: port}},
		}
		return port
	})
}

func TestEnabledEngineRestorationSurvivesInvalidSettingsJournal(t *testing.T) {
	b := &Broker{clusterDir: filepath.Join(t.TempDir(), "cluster")}
	path, err := b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{invalid"), 0600); err != nil {
		t.Fatal(err)
	}
	worker, codec := newTestRPCWorkerPipe(t)
	restored := make(chan string, 1)
	go func() {
		msg, err := codec.Read()
		if err != nil {
			t.Error(err)
			return
		}
		restored <- msg.Method
	}()
	b.restoreEnabledEngines(worker)
	select {
	case method := <-restored:
		if method != restoreEnabledEnginesMethod {
			t.Fatalf("method=%q", method)
		}
	case <-time.After(time.Second):
		t.Fatal("invalid journal suppressed enabled-engine restoration")
	}
	if b.engineSettingsError == nil {
		t.Fatal("invalid journal was not reported")
	}
}

func TestSettingsFullCommandJournalMigratesBeforeRecovery(t *testing.T) {
	h := newSettingsHarness(t)
	request := h.request(t)
	h.b.engineConfigMu.Lock()
	record := h.b.engineSettings["ollama"]
	record.Snapshot.Format = "pair-launch-v1"
	record.Snapshot.Settings.LaunchText = "managed serve --fixture-option --future-option"
	record.Snapshot.Phase = "applying"
	record.Resume = true
	oldProxyPort := record.Snapshot.Settings.ProxyPort
	if err := h.b.saveEngineSettingsLocked(); err != nil {
		t.Fatal(err)
	}
	h.b.engineConfigMu.Unlock()
	if !h.b.recoverEngineSettings() {
		t.Fatal("recovery failed")
	}
	snapshot, err := h.b.getEngineSettings(context.Background(), settings.Request{Engine: "ollama"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Format != "pair-arguments-v1" || snapshot.Phase != "succeeded" || snapshot.Settings.LaunchText != "--fixture-option --future-option" || snapshot.Settings.ProxyPort != oldProxyPort || snapshot.Settings.ServerPort != request.Settings.ServerPort {
		t.Fatalf("migration lost accepted configuration: %+v", snapshot)
	}
	if h.applies.Load() != 1 {
		t.Fatal("pending operation was not applied exactly once")
	}
}
