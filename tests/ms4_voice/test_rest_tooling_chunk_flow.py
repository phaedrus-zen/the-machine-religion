from __future__ import annotations

import json

from machine_spirit_4.gateway.spoken_text_filter import sanitize_for_speech
from machine_spirit_4.gateway.voice import SentenceChunker


def _new_chunker() -> SentenceChunker:
    return SentenceChunker(
        first_chunk_min_words=5,
        min_chunk_words=10,
        max_chunk_words=40,
    )


def _prime_startup_pair(chunker: SentenceChunker) -> None:
    chunks = chunker.add(
        "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 "
    )
    assert [len(chunk.split()) for chunk in chunks] == [5, 10]


def _physical_gpu_tool_reply() -> str:
    payload = {
        "gpus": [
            {
                "assigned_to": ["TTS_SUPER"],
                "backend": "gim",
                "index": "0",
                "lan_available": True,
                "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "vram_free_mb": 92812,
                "vram_source": "detailed_gpu_info",
                "vram_total_mb": 97887,
                "vram_used_mb": 3777,
            },
            {
                "assigned_to": ["SD", "TTS_SUPER"],
                "backend": "gim",
                "index": "1",
                "lan_available": True,
                "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "vram_free_mb": 96570,
                "vram_source": "detailed_gpu_info",
                "vram_total_mb": 97887,
                "vram_used_mb": 19,
            },
        ],
        "node_mode": "Cluster",
    }
    return (
        "Authoritative result from hivemind.gpu.availability@v1: "
        + json.dumps(payload)
    )


def test_physical_gpu_tool_reply_coalesces_at_clause_boundaries_after_startup_pair():
    reply = _physical_gpu_tool_reply()
    assert len(reply) == 600

    chunker = _new_chunker()
    chunker.first_chunk_sanitizer = sanitize_for_speech
    chunks = chunker.add(reply)
    tail = chunker.flush()
    if tail:
        chunks.append(tail)

    # Preserve the measured 5/10 startup runway: a short onset chunk followed
    # by enough speech to cover synthesis of the next clause.
    assert chunks[:2] == [
        'Authoritative result from hivemind.gpu.availability@v1: {"gpus": [{"assigned_to"',
        '["TTS_SUPER"], "backend": "gim", "index": "0", "lan_available": true, "name": "NVIDIA RTX',
    ]
    # Once the startup pair is scheduled, use the existing ten-word coalescing
    # floor at a real clause delimiter instead of falling through to one
    # 40-word / 418-character head-of-line chunk.
    assert chunks[2] == (
        'PRO 6000 Blackwell Workstation Edition", "vram_free_mb": 92812, '
        '"vram_source": "detailed_gpu_info", "vram_total_mb": 97887,'
    )
    assert [(len(chunk.split()), len(chunk)) for chunk in chunks] == [
        (6, 80),
        (10, 89),
        (11, 123),
        (11, 114),
        (10, 83),
        (8, 105),
    ]


def test_startup_second_chunk_preserves_runway_before_late_weak_boundary():
    chunker = _new_chunker()
    words = [f"w{index}" for index in range(1, 21)]
    words[-1] += ","

    chunks = chunker.add(" ".join(words) + " ")

    assert [len(chunk.split()) for chunk in chunks] == [5, 10]
    assert chunks[:2] == [
        "w1 w2 w3 w4 w5",
        "w6 w7 w8 w9 w10 w11 w12 w13 w14 w15",
    ]
    assert chunker.buffer.split() == words[15:]


def test_steady_chunk_prefers_hard_ceiling_before_late_weak_boundary():
    chunker = _new_chunker()
    _prime_startup_pair(chunker)
    words = [f"w{index}" for index in range(1, 46)]
    words[-1] += ","

    chunks = chunker.add(" ".join(words) + " ")

    assert [len(chunk.split()) for chunk in chunks] == [40]
    assert chunker.buffer.split() == words[40:]


def test_steady_chunk_does_not_split_unfinished_colon_continuations():
    cases = [
        (
            "Visit the secure documentation mirror now using the address https:",
            "//example.com before continuing with the remaining spoken reply. ",
            "https://example.com",
        ),
        (
            'Read this structured response now and preserve the next JSON "field":',
            ' "value", before continuing with the remaining spoken reply. ',
            '"field": "value",',
        ),
    ]

    for prefix, suffix, expected_continuation in cases:
        chunker = _new_chunker()
        _prime_startup_pair(chunker)

        assert chunker.add(prefix) == []
        chunks = chunker.add(suffix)
        tail = chunker.flush()
        if tail:
            chunks.append(tail)

        assert expected_continuation in " ".join(chunks)
