// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react'
import { type EngineType } from '@/shared/types/engines'
import ollamaIcon from '@/ui/assets/engine-icons/ollama.png?inline'
import lmStudioIcon from '@/ui/assets/engine-icons/lm-studio.png?inline'

export default function EngineIcon({ type, size = 32 }: { type: EngineType; size?: number }) {
    const dimension = `${size}px`
    const imgStyle: CSSProperties = { width: '100%', height: '100%', objectFit: 'contain' }
    const containerStyle: CSSProperties = {
        width: dimension,
        minWidth: dimension,
        maxWidth: dimension,
        height: dimension,
        minHeight: dimension,
        maxHeight: dimension,
        backgroundColor: '#fff',
        borderRadius: '25%',
        overflow: 'hidden'
    }

    if (type === 'ollama') {
        return (
            <div style={containerStyle}>
                <img src={ollamaIcon} alt="Ollama" style={imgStyle} />
            </div>
        )
    }

    if (type === 'lm-studio') {
        imgStyle.objectFit = 'cover'

        return (
            <div style={containerStyle}>
                <img src={lmStudioIcon} alt="LM Studio" style={imgStyle} />
            </div>
        )
    }

    // The user-managed engine has no vendor to brand: a generic server glyph.
    if (type === 'openai-compatible') {
        return (
            <div style={containerStyle}>
                <svg
                    viewBox="0 0 24 24"
                    width="100%"
                    height="100%"
                    role="img"
                    aria-label="OpenAI-compatible server"
                >
                    <g fill="none" stroke="#2f2f2f" strokeWidth="1.6">
                        <rect x="4" y="4.5" width="16" height="6.4" rx="1.6" />
                        <rect x="4" y="13.1" width="16" height="6.4" rx="1.6" />
                    </g>
                    <circle cx="7.4" cy="7.7" r="1.15" fill="#2f2f2f" />
                    <circle cx="7.4" cy="16.3" r="1.15" fill="#2f2f2f" />
                </svg>
            </div>
        )
    }

    return null
}
