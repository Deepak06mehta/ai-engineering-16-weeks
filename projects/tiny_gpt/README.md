# Project 1 — Tiny GPT from Scratch  (Week 4)

Vault note: `Deepak_Library/AI Engineer/06-Projects/Project 1 - Tiny GPT from Scratch`

An ~11M-parameter decoder-only transformer written from an empty file. It will be bad
at generating text; the point is that you built every piece and can explain every line.

## Target config
```python
vocab_size=8_000, d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
head_dim=64, d_ff=1024, max_seq_len=256, rope_theta=10_000.0, qk_norm=True
```

## The five tests that matter
- `test_loss_at_init_is_ln_vocab` — within 2% of `ln(V)`
- `test_no_future_leakage` — change token t+1, output at t must be bit-identical
- `test_single_batch_overfit` — 300 steps on ONE batch, loss < 0.05
- `test_rope_is_relative` — `score(q_m, k_n)` depends only on `m - n`
- `test_generation_respects_context_limit`

## Status
- [ ] not started
