# ai-engineering-16-weeks

A 16-week programme to move from backend engineering to Applied AI / LLM engineering:
from-scratch transformer components, RAG, agents, fine-tuning, evaluation and deployment.

**Max 3 hours per study day · 6 study days per week · 1 rest day.**

The notes, daily schedule and architecture reference live in the Obsidian vault at
`Deepak_Library/AI Engineer/` — start at `_AI Engineer Knowledge Map`.
This repository is where the **code** goes.

---

## First-day setup

```bash
make check     # what's installed, what's missing, and the exact next command
```

```bash
make setup     # create .venv on Python 3.12 and install Weeks 1-6 dependencies
```

```bash
source .venv/bin/activate && make test
```

```bash
cp .env.example .env
```

Then pull the three local models you will use all programme:

```bash
ollama pull qwen3:4b && ollama pull llama3.2:3b && ollama pull nomic-embed-text
```

Three families on purpose — Qwen (QK-Norm, dense), Llama (the classic GQA baseline),
and an embedding model so Week 10 costs nothing.

---

## Daily commands

| Command | When |
|---|---|
| `make check` | Week 1 Day 1, and whenever something stops working |
| `make test` | every day, end of Block D — must stay green |
| `make exercises` | the work you have not done yet — **red is correct here** |
| `make lint` | before every commit |
| `make up` / `make down` | Weeks 10–16, the data services |
| `make setup-models` | start of Week 7 |
| `make setup-apps` | start of Week 9 |

---

## How the tests are split

```
tests/              # scaffold + environment. ALWAYS GREEN.
tests/exercises/    # marked `exercise`. RED until you implement them. That is the point.
```

Exercise tests are deselected by default so `make test` is a meaningful health check.
Run them deliberately with `make exercises`.

Two are already waiting for you:

| Exercise | Week | What it checks |
|---|---|---|
| `test_week01_tokenizer.py` | 1, Wed | round-trip, empty string, unknown char, whitespace runs, unicode, determinism |
| `test_week06_kv_calc.py` | 6, Tue | your KV calculator must reproduce **six published figures** from the LLM Architecture Gallery |

The Week 6 one is the most valuable test in the repository. If your number disagrees
with the gallery's, your model of that architecture is wrong — finding out which
assumption broke *is* the exercise.

---

## Layout

```
.
├── scripts/env_check.py      # make check
├── src/
│   ├── common/               # device.py, seeds.py, timing.py, kv_calc.py
│   └── week01_tokenizer/     # Week 1 exercise
├── tests/                    # scaffold (green) + exercises/ (red until done)
├── notes/                    # daily logs, weekly log, your own concept notes
├── notebooks/                # attention-from-scratch, moe, kv-cache ...
├── benchmarks/               # one JSON + one markdown report per benchmark run
├── projects/
│   ├── tiny_gpt/             # Project 1  (Week 4)
│   ├── finetune_lab/         # Project C  (Week 8)
│   ├── rag_assistant/        # Project A  (Weeks 10-14)
│   └── devops_assistant/     # Project B  (Week 12)
├── infra/                    # docker-compose + pgvector schema
└── requirements/             # core.txt (W1-6), models.txt (W7-8), apps.txt (W9-16)
```

---

## Rules that make this work

1. **Commit every study day**, even if it is one failing test and a note. The commit
   graph is the portfolio artefact a hiring manager actually reads.
2. **Never commit weights or source documents.** Commit the ingestion script, not the
   corpus. `.gitignore` enforces this and `tests/test_scaffold.py` checks it.
3. **Record hardware, runtime, quantisation, context length and batch for every
   benchmark.** A number without its setup is noise.
4. **Predict before you run.** Write down the tensor shape you expect, then check.
5. **Debug on one batch.** If a model cannot memorise a single batch, it has a bug.
   Never debug on the full dataset.

Commit message format:

```
w04d03: single-batch overfit for tiny GPT reaches loss 0.03

Confirms the training loop and loss wiring are correct before touching
the real dataset. LR 3e-4, 200 steps, seed 1337.
```

---

## Known constraints on this machine

- **Apple Silicon → MPS**, not CUDA. No `float64`, no FlashAttention-2, no multi-GPU.
  Everything in Weeks 1–6 still runs in minutes.
- **`bitsandbytes` 4-bit (QLoRA) is CUDA-only.** Week 8 needs either a ≤1B model in
  bf16 locally, or 1–2 hours of rented GPU. Plan it on the Monday, not the Thursday.
- **Docker is not installed yet.** Needed from Week 10: `brew install --cask orbstack`.
- **`pdftotext` (poppler)** is needed from Week 10: `brew install poppler`.
