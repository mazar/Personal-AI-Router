// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'

describe('remote engine status', () => {
    it('prefers authoritative stopped facts over proxy presence', () => {
        const state = getModularBridgeState()
        const remoteNodeId = 'remote-engine-status-remote'

        state.setSelfId('remote-engine-status-local')
        state.handleNotification({
            source: 'lmstudio-proxy',
            method: 'node/discovered',
            params: {
                id: remoteNodeId,
                host: remoteNodeId,
                port: 1234,
                addresses: ['192.0.2.190'],
                ip: '192.0.2.190'
            }
        })
        expect(state.isRemoteEngineRunning(remoteNodeId, 'lm-studio')).toBe(true)

        state.applyRemoteEngineFacts(remoteNodeId, {
            engines: [
                {
                    engine: 'lmstudio',
                    installed: true,
                    running: false,
                    healthy: false,
                    port: 1235
                }
            ]
        })

        expect(state.isRemoteEngineRunning(remoteNodeId, 'lm-studio')).toBe(false)
        expect(
            state
                .getEngineInitialState()
                .statuses.find(
                    status => status.nodeId === remoteNodeId && status.engineType === 'lm-studio'
                )
        ).toMatchObject({
            processStatus: 'stopped',
            enginePort: 1235,
            // The proxy remains discoverable even though the engine behind it is stopped.
            proxyPort: 1234
        })
    })

    it('surfaces a clustered peer user-managed engine from facts and attribution', () => {
        const state = getModularBridgeState()
        const remoteNodeId = 'remote-rider-status-remote'

        state.setSelfId('remote-rider-status-local')
        // The peer's broker discovery carries per-engine attribution; its ollama
        // proxy presence advertises the facade the rider's models ride.
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: remoteNodeId,
                        name: 'peer',
                        ipAddress: '192.0.2.191',
                        port: 14322,
                        clustered: true,
                        trusted: true,
                        modelsByEngine: { 'openai-compatible': ['vllm-model'] }
                    }
                ]
            }
        })
        state.handleNotification({
            source: 'ollama-proxy',
            method: 'node/discovered',
            params: {
                id: remoteNodeId,
                host: remoteNodeId,
                port: 11434,
                addresses: ['192.0.2.191'],
                ip: '192.0.2.191'
            }
        })

        // No facts yet: the facade's advertisement cannot tell whether the
        // rider is the engine that is up, so a clustered peer's user-managed
        // engine asserts no state.
        expect(
            state
                .getEngineInitialState()
                .statuses.find(
                    status =>
                        status.nodeId === remoteNodeId && status.engineType === 'openai-compatible'
                )
        ).toBeUndefined()

        state.applyRemoteEngineFacts(remoteNodeId, {
            engines: [
                {
                    engine: 'openai-compatible',
                    installed: true,
                    running: true,
                    healthy: true,
                    port: 8888
                }
            ]
        })

        expect(
            state
                .getEngineInitialState()
                .statuses.find(
                    status =>
                        status.nodeId === remoteNodeId && status.engineType === 'openai-compatible'
                )
        ).toMatchObject({
            processStatus: 'running',
            enginePort: 8888,
            // The rider's endpoint is the facade engine's advertised proxy port.
            proxyPort: 11434
        })

        // The peer's per-engine attribution is authoritative: the rider's model
        // renders under the user-managed engine, attributed to it alone.
        const riderModels = state
            .getEngineInitialState()
            .models.find(
                entry => entry.nodeId === remoteNodeId && entry.engineType === 'openai-compatible'
            )
        expect(riderModels?.models.map(model => model.name)).toEqual(['vllm-model'])
    })
})
