"""WEEK 6, TUESDAY — KV-cache calculator. Run with: make exercises

The expected values are the LLM Architecture Gallery's published KiB/token figures.
If your calculator disagrees with one of them, your model of that architecture is
wrong. Find out which assumption broke — that search is the exercise.

Source: https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
"""

from __future__ import annotations

import pytest

from src.common.kv_calc import AttnLayer, kv_bytes_per_token, kv_total_bytes, uniform_stack

pytestmark = pytest.mark.exercise

KIB = 1024


@pytest.mark.parametrize(
    ("name", "n_layers", "n_kv_heads", "head_dim", "expected_kib"),
    [
        # model                layers  kv_heads  head_dim  published KiB/token (bf16)
        ("GPT-2 XL 1.5B (MHA)",     48,       25,      64,  300),
        ("Llama 3 8B (GQA 32/8)",   32,        8,     128,  128),
        ("Qwen3 8B (GQA)",          36,        8,     128,  144),
        ("Qwen3 32B (GQA)",         64,        8,     128,  256),
        ("OLMo 2 7B (MHA)",         32,       32,     128,  512),
        ("Llama 3.2 1B (GQA)",      16,        8,      64,   32),
    ],
)
def test_matches_published_figures(name, n_layers, n_kv_heads, head_dim, expected_kib):
    got = kv_bytes_per_token(uniform_stack(n_layers, n_kv_heads, head_dim))
    assert got == expected_kib * KIB, f"{name}: got {got / KIB:.1f} KiB, expected {expected_kib}"


def test_gqa_is_cheaper_than_mha_at_equal_depth():
    mha = kv_bytes_per_token(uniform_stack(32, 32, 128))
    gqa = kv_bytes_per_token(uniform_stack(32, 8, 128))
    assert gqa * 4 == mha


def test_cross_layer_sharing_reduces_producing_layers():
    """Gemma 4 E2B style: only some layers write to the cache."""
    full = [AttnLayer(n_kv_heads=1, head_dim=256) for _ in range(8)]
    shared = [
        AttnLayer(n_kv_heads=1, head_dim=256, shares_with_earlier_layer=(i % 2 == 1))
        for i in range(8)
    ]
    assert kv_bytes_per_token(shared) == kv_bytes_per_token(full) // 2


def test_total_scales_with_context_and_batch():
    stack = uniform_stack(32, 8, 128)
    per_token = kv_bytes_per_token(stack)
    assert kv_total_bytes(stack, seq_len=1000, batch_size=4) == per_token * 1000 * 4


def test_llama3_8b_at_128k_is_about_16_gib():
    """The number that explains why GQA, MLA and sliding windows all exist."""
    total = kv_total_bytes(uniform_stack(32, 8, 128), seq_len=131_072)
    assert 15.5 < total / 1024**3 < 16.5
