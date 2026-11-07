# Project C — Fine-Tuning and Evaluation Lab  (Week 8)

Vault note: `Deepak_Library/AI Engineer/06-Projects/Project C - Fine-Tuning Lab`

Not a product — a **reproducible experiment** answering one question honestly:
*did fine-tuning a small model on my data actually help?*

"No, RAG would have been better" is a perfectly good result and a stronger portfolio
piece than an unexamined yes.

> **Hardware warning:** `bitsandbytes` 4-bit (QLoRA) is CUDA-only and will NOT run on
> Apple Silicon. Design the whole experiment locally, then rent a GPU for 1–2 hours.
> See the vault note "Cloud GPU Escape Hatch".

## Order that matters
1. dataset + held-out split (created FIRST)
2. dataset card
3. **baseline evaluation BEFORE any training**
4. LoRA run, every hyperparameter recorded including the seed
5. same 30 prompts after, temperature 0
6. find five regressions and write them up
7. a verdict

## Status
- [ ] not started
