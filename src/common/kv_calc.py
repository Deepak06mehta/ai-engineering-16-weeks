"""KV-cache calculator — WEEK 6 EXERCISE. The bodies are yours to write.

Why this matters: roughly half of every modern LLM architecture decision exists to
make the KV cache smaller. Building this calculator is what turns that from a
sentence you have read into something you understand.

The formula, for one standard attention layer in bf16 (2 bytes per element):

    bytes_per_layer_per_token = cached_tensors * n_kv_heads * head_dim * bytes_per_elem
                              = 2 * n_kv_heads * head_dim * 2

Your implementation must reproduce the published figures in tests/exercises/
test_week06_kv_calc.py. If a number disagrees, your model of that architecture is
wrong — find out which assumption broke. That search IS the exercise.

Run it when done:
    python -m src.common.kv_calc --layers 32 --kv-heads 8 --head-dim 128 --seq-len 131072
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

BYTES_BF16 = 2


@dataclass
class AttnLayer:
    """One cache-producing attention layer."""

    n_kv_heads: int
    head_dim: int
    cached_tensors: int = 2  # K and V
    shares_with_earlier_layer: bool = False  # cross-layer KV sharing (Gemma 4 E2B/E4B)


def kv_bytes_per_token(layers: list[AttnLayer], bytes_per_elem: int = BYTES_BF16) -> int:
    """Logical cache added by ONE token, batch 1.

    Only layers that actually produce new K/V contribute — a layer that reuses an
    earlier layer's tensors adds nothing to the cache.
    """
    raise NotImplementedError("Week 6, Tuesday. Write this yourself.")


def kv_total_bytes(
    layers: list[AttnLayer],
    seq_len: int,
    batch_size: int = 1,
    bytes_per_elem: int = BYTES_BF16,
) -> int:
    """Total logical cache for a given context length and batch size."""
    raise NotImplementedError("Week 6, Tuesday. Write this yourself.")


def uniform_stack(n_layers: int, n_kv_heads: int, head_dim: int) -> list[AttnLayer]:
    """Helper: a model where every layer has the same attention shape."""
    return [AttnLayer(n_kv_heads=n_kv_heads, head_dim=head_dim) for _ in range(n_layers)]


def human(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n_bytes) < 1024 or unit == "TiB":
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description="KV cache calculator (Week 6)")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()

    stack = uniform_stack(a.layers, a.kv_heads, a.head_dim)
    per_token = kv_bytes_per_token(stack)
    total = kv_total_bytes(stack, a.seq_len, a.batch)
    print(f"per token : {human(per_token)}")
    print(f"at {a.seq_len:,} tokens x batch {a.batch}: {human(total)}")


if __name__ == "__main__":
    main()
