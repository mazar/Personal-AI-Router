// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"reflect"
	"testing"
)

// A successful engine:models sweep is authoritative about the inventories it
// names: an engine reported with an empty list is running and serving no
// models, so its cached inventory must leave the local-backend payloads — a
// model the user deleted stops being routed to this node. An engine the sweep
// could not query is absent from the result and keeps its cached list; a
// failed sweep changes nothing at all.
func TestRefreshEngineModelsEvictsAnAuthoritativelyEmptyInventory(t *testing.T) {
	payloads := make(chan map[string]any, 2)
	worker, workerCodec := newTestRPCWorkerPipe(t)
	b := &Broker{codec: NewCodec(&bytes.Buffer{})}
	b.engineModels = map[string][]string{
		"ollama":            {"qwen3", "llama3"},
		"openai-compatible": {"vllm-model"},
	}
	b.setEngineMgr(worker)
	go func() {
		codec := workerCodec
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if !msg.IsRequest() || msg.Method != "engine:models" {
				continue
			}
			payload, ok := <-payloads
			if !ok {
				return
			}
			_ = codec.Respond(msg.ID, payload)
		}
	}()

	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()

	// Ollama swept to a smaller set and the rider to an authoritative empty:
	// the stale ollama entries are overwritten and the rider's cache is evicted.
	payloads <- map[string]any{"modelsByEngine": map[string]any{
		"ollama":            []string{"qwen3"},
		"openai-compatible": []string{},
	}}
	b.refreshEngineModelsLocked()
	if got := b.cachedEngineModels("openai-compatible"); got != nil {
		t.Errorf("cached rider inventory after an authoritative empty sweep = %v, want evicted", got)
	}
	if got := b.cachedEngineModels("ollama"); !reflect.DeepEqual(got, []string{"qwen3"}) {
		t.Errorf("cached ollama inventory = %v, want the swept [qwen3]", got)
	}

	// A sweep that does not name an engine says nothing about it: the rider's
	// absence here is "not queryable", not "empty", so nothing changes.
	payloads <- map[string]any{"modelsByEngine": map[string]any{
		"ollama": []string{"qwen3"},
	}}
	b.refreshEngineModelsLocked()
	if got := b.cachedEngineModels("openai-compatible"); got != nil {
		t.Errorf("cached rider inventory after a sweep without the engine = %v, want still evicted", got)
	}
	if got := b.cachedEngineModels("ollama"); !reflect.DeepEqual(got, []string{"qwen3"}) {
		t.Errorf("cached ollama inventory after a sweep without it = %v, want kept", got)
	}
}
