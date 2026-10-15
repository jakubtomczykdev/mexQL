# MexQL

Repository: https://github.com/jakubtomczykdev/MexQL

Text-to-SQL on the [BIRD](https://bird-bench.github.io/) benchmark with a small open LLM
fine-tuned by reinforcement learning (GRPO with an execution-based reward), plus a variant
for questions asked in Polish.

Project stages:

1. **Environment, evaluation and baselines** (this repository state)
2. Filtering of training questions by the base model's pass rate
3. Optional short SFT (rejection sampling), then GRPO with an execution reward
4. Final evaluation, ablations, error analysis
5. Polish questions and a Gradio demo on Hugging Face

## What is here

| Component | Where | Notes |
|---|---|---|
| Data download / unpacking | `scripts/download_data.py`, `src/mexql/data/download.py` | official `dev.zip` / `train.zip`; `--train-meta-only` fetches only the 7 MB of train metadata through HTTP range requests |
| Splits | `src/mexql/data/bird.py` | `val_internal` = 500 train questions, stratified by database, seeded; `train_rl` = the rest; `dev` is used only for final evaluation |
| Prompt | `src/mexql/schema.py`, `src/mexql/prompt.py` | `CREATE TABLE` statements rebuilt from `PRAGMA` (types, primary and foreign keys), up to 3 distinct example values per column, evidence, question |
| Response format | `src/mexql/parsing.py` | `<think>…</think>` followed by one ```` ```sql ```` block. SQL extraction is lenient (last `sql` block anywhere), strict compliance is reported separately as `format_ok` / `format_issue`; unparsable output gets `format_error` instead of an exception |
| SQL execution | `src/mexql/execution/` | SQLite opened with `mode=ro` plus an authorizer (no ATTACH/PRAGMA), per-query timeout (SQLite progress handler + process watchdog), row cap (`too_large`), order-insensitive row digests instead of materialised results, gold result cache, multiprocessing |
| Metric | `src/mexql/metrics.py` | execution accuracy (EX) with the official set-equality semantics; per difficulty; failure breakdown |
| Inference | `src/mexql/inference/` | `vllm` (Linux + NVIDIA, batched, prefix caching), `hf` (transformers on CUDA / Apple MPS / CPU, for smoke tests), `dummy` (no model) |
| Entry points | `scripts/evaluate.py`, `scripts/sample_train.py`, `scripts/prepare_data.py`, `scripts/export_official.py` | every script has `--smoke` (20 questions) |
| Colab | `scripts/colab_bootstrap.sh`, `notebooks/colab_mexql.ipynb` | GPU runs happen on Google Colab |
| Tests | `tests/` | pytest, including `test_official_compat.py`, which checks our verdicts against the unmodified official `evaluation.py` (vendored in `tests/vendor/`) |

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/). Code, configuration and outputs are English;
`data/` and `outputs/` are git-ignored.

```bash
uv sync --extra hf --extra dev          # laptop: transformers backend + tests
uv sync --extra hf --extra gpu --extra dev   # Linux + NVIDIA: adds vLLM
```

On Colab run `bash scripts/colab_bootstrap.sh` (or open `notebooks/colab_mexql.ipynb`); it
installs vLLM with `uv pip install ... --torch-backend=auto`, which picks the PyTorch build for
the session's CUDA driver.

Data:

```bash
uv run python scripts/download_data.py --dev --train-meta-only   # ~2 GB on disk
uv run python scripts/download_data.py --train                   # ~9 GB download, ~32 GB unpacked
uv run python scripts/prepare_data.py --model configs/models/qwen3_4b_instruct.yaml
```

`prepare_data.py` builds the split, the schema cache, the gold result cache and a prompt-length
report (`outputs/reports/prompt_lengths_<model>.md`). Sources whose databases are not present are
skipped, so the dev set alone is enough on a laptop.

## Running an evaluation

One command evaluates any model on any split with greedy decoding and prints EX by difficulty:

```bash
uv run python scripts/evaluate.py --model configs/models/qwen3_4b_instruct.yaml --split dev
uv run python scripts/evaluate.py --model configs/models/qwen25_coder_3b.yaml --split val_internal
uv run python scripts/evaluate.py --model configs/models/qwen25_coder_0.5b.yaml --backend hf --smoke
```

Any config value can be overridden: `--set sampling.max_tokens=512 --set execution.timeout_s=60`.

Long runs on preemptible machines (Colab) use `--chunk-size N --resume --sync-dir <dir>`: results
are appended after every chunk of N questions, the run directory is mirrored to `<dir>` (for
example a mounted Google Drive) after each chunk, and a restarted command restores the mirror and
continues with the questions that are not in the output file yet. The same flags exist on
`scripts/sample_train.py`.

Outputs land in `outputs/runs/<run_name>/`:

* `predictions.jsonl`: one row per question with `prompt`, `response`, extracted `sql`,
  `parse_status`, `format_ok` / `format_issue`, `exec_status`, final `status` and `correct`
* `metrics.json`, `metrics.md`: EX per difficulty, the status breakdown and the strict
  format-compliance rate (closed `<think>` block followed by exactly one `sql` block)

Final statuses: `correct`, `wrong_result`, `format_error`, `sql_error`, `timeout`, `too_large`,
`prompt_too_long`, `gold_invalid` (the gold query itself fails; excluded from `ex`, counted as
wrong in `ex_all`, which is the official-script-compatible number).

To cross-check a dev run with the upstream script:

```bash
uv run python scripts/export_official.py --run outputs/runs/qwen3-4b-instruct-2507_dev
```

## Stage-2 preparation: pass rates on the train set

```bash
uv run python scripts/sample_train.py --model configs/models/qwen3_4b_instruct.yaml --limit 200
```

samples 8 completions per question at temperature 0.8, executes all of them and writes
`pass_rates.jsonl` (per question: `n_correct`, `pass_rate`, statuses, SQLs) and `summary.json`
with a histogram of correct samples and a time estimate for the whole `train_rl` split
(extrapolated from the questions generated in the current session). `--limit 0` processes the
whole split; combine it with `--chunk-size 100 --resume --sync-dir ...` on Colab.

## Results

### Stage-1 status

The full greedy baselines on `dev` (1534 questions) and `val_internal` (500 questions) and the
stage-2 sampling run are executed on Google Colab (`notebooks/colab_mexql.ipynb`); this machine
(Apple M4, no CUDA) only runs the 20-question smoke tests through the `hf` backend. Numbers
below are therefore smoke-test numbers; the Colab rows are filled in from
`python scripts/collect_results.py` after the notebook has run.

### Smoke tests (20 dev questions spread over all 11 databases, greedy, Apple M4 / MPS)

| model | n | simple | moderate | challenging | EX | format ok | status breakdown | gen. time |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 20 | 69.2 (9/13) | 33.3 (1/3) | 75.0 (3/4) | **65.0** | 100% | correct 13, wrong_result 6, sql_error 1 | 34 min |
| Qwen2.5-Coder-0.5B-Instruct (pipeline check only) | 20 | 23.1 | 0.0 | 0.0 | **15.0** | 40% | correct 3, wrong_result 5, sql_error 12 | 4 min |

Qwen3-4B answers with a median of 172 completion tokens and followed the `<think>` + single
`sql` block format in 20/20 cases. (This only became true after the think tags in the prompt
were re-encoded as plain text; fed as Qwen3's added tokens `<think>` / `</think>` the model
answered with a bare `<tool_call>` in 15/20 cases, see `mexql.prompt.escaped_special_tags`.)

### Full baselines (Colab, to be filled in)

| model | split | n | simple | moderate | challenging | EX | EX (all) | format ok |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | dev | 1534 | – | – | – | – | – | – |
| Qwen3-4B-Instruct-2507 | val_internal | 500 | – | – | – | – | – | – |
| Qwen2.5-Coder-3B-Instruct | dev | 1534 | – | – | – | – | – | – |

`EX` excludes questions whose gold query does not execute within 30 s, `EX (all)` counts them as
wrong (this is what the official script reports). On dev two gold queries time out on this
machine (question ids 518 and 701; 518 finishes in 47 s, 701 does not finish within 150 s), so
`EX (all)` is at most 99.87.

### Data and prompt statistics (dev, Qwen3 tokenizer)

| | value |
|---|---|
| questions | 1534 (925 simple / 464 moderate / 145 challenging) |
| prompt tokens | min 569, median 2658, p90 3903, p99 5356, max 5429 |
| longest database | european_football_2 (5282–5429 tokens per prompt) |
| prompts over the 15 360-token budget (16k window, 1k generation) | 0 |
| largest gold result | 278 230 rows |

Train (`train.json`): 9428 questions over 69 databases, no difficulty labels; `val_internal`
holds 500 of them (seed 1234, stratified by database), `train_rl` the remaining 8928.

### Stage-2 time estimate (8 samples per question, T = 0.8, `train_rl` = 8928 questions)

`scripts/sample_train.py --limit 200` prints `estimate_full_split` extrapolated from the
measured 200-question run; that is the number to put here once it has run on Colab. Until
then, an a-priori estimate (assumption, not a measurement): with vLLM, prefix caching (the 8
samples share one prompt) and ~200 generated tokens per sample, the run is decode-bound at
roughly 1.6k tokens per question, i.e. about 1–1.5 h on an L4 or A100 and 4–6 h on a T4
(fp16, no FlashAttention, several Colab sessions). Executing the ~71k sampled queries takes
well under an hour on Colab's CPUs (dev queries execute in milliseconds, the slowest in a few
seconds, each capped at 30 s).

### Sanity check against the official script

`tests/test_official_compat.py` compares verdicts on 47 synthetic pairs and 132 BIRD-dev pairs
(22 gold queries x 6 perturbations); `scripts/export_official.py` writes `predict_dev.json`
for a dev run so the official `evaluation.py` can be run on the same predictions.

## Evaluation semantics and known differences from the official script

Our verdicts agree with the official `evaluation.py` on every synthetic and BIRD-dev pair in
`tests/test_official_compat.py`. Deliberate differences:

* The official script executes the predicted and the gold query inside one 30 s window; we give
  each query its own 30 s and cache gold results, so a pair where both queries take 20 s passes
  here and times out there.
* The official script executes any statement on a writable connection; we refuse writes,
  `ATTACH` and `PRAGMA` (read-only mode plus an authorizer) and treat them as `sql_error`.
* Result sets above `execution.max_rows` (500 000) are scored as `too_large` (wrong); the
  official script has no cap. The largest gold result on dev has 278 230 rows.
* Row sets are compared through 16-byte BLAKE2 digests of canonicalised rows instead of
  Python sets of tuples; equality semantics (`1 == 1.0`, `'1' != 1`, `NULL`) are preserved.

## Tests

```bash
uv run pytest -q          # ~40 s with the dev set present, skips BIRD-dependent tests otherwise
```
