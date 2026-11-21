# TS_T5 release: refactor, Hugging Face models, published results

Date: 2026-11-20
Status: reviewed, awaiting author approval

## Intent

Turn this repository into the reference companion for the INLG 2021 paper
*Controllable Sentence Simplification with a Unified Text-to-Text Transfer
Transformer* (Sheang & Saggion). Goal: **reuse and citation**. A researcher
should be able to load the paper's models from Hugging Face in three lines,
reproduce the paper's TurkCorpus and ASSET scores with one command, and read
what was tried and learned without opening the old drives.

## Goals

1. Publish the two paper models on Hugging Face:
   `kcsheang/ts-t5-base-4tokens` and `kcsheang/ts-t5-base-5tokens`, each with a
   full model card, plus an HF Collection linking both and the paper.
2. Refactor the code to run on current Python / PyTorch / transformers /
   Lightning, packaged with `pyproject.toml` and a small CLI
   (`train`, `evaluate`, `simplify`).
3. Reproduce the README numbers with the refactored code before any upload
   (gate below).
4. Document experiments, results, ablations and findings in `docs/`.
5. Publish the human evaluation: protocol, aggregate scores and per-item
   ratings with worker IDs pseudonymised.
6. License everything CC BY-NC 4.0.

## Non-goals

- New experiments, new models, or improving on the paper's scores.
- Publishing the ~30 exploratory runs as checkpoints (summarised in docs only).
- The t5-small run (not in the paper, scores unverified).
- An HF dataset repo or Space (results live in GitHub `docs/`).
- Support for the multilingual / other-architecture sibling projects on D8
  (`TS_mT5`, `TS_Bart`, ...).

## Anti-goals (failure even if it "works")

- Published models whose scores do not match what the README and paper claim.
- Any *new* Mechanical Turk worker ID or other personal data in a public repo.
  (The raw exports already tracked under `Human evaluation/results/raw from
  mTurk/` stay as they are, by the author's decision; history is not
  rewritten.)
- Changed generation behaviour (decoding settings, control-token format)
  without that change being documented as such.
- A release that cannot be installed with `pip install -e .` on a clean
  machine.

## Current state (what the builder starts from)

### Repo (`/Users/kim/Sync/workspace/TS_T5_github`, GitHub `KimChengSHEANG/TS_T5`)

- `source/` (~1.9k lines): `preprocessor.py` (control-token features, adopted
  from facebookresearch/access), `model.py` (`T5FineTuner` LightningModule +
  datasets), `train.py`, `evaluate.py` (module-level globals, beam search
  `num_beams=8`), `find_best_tokens.py` / `hyperparameters_tuning.py` (Optuna
  token search), `download_data.py` (gdown from Google Drive),
  `resources.py`, `helper.py`, `utils.py`.
- `scripts/`: thin wrappers, one per use case (several near-duplicates).
- `resources/`: datasets (WikiLarge, Wiki-Auto, TurkCorpus, ASSET),
  `DUMPS/` (GloVe 42B pickle, preprocessor pickle), processed-data caches.
- `system_outputs/`: the two models' TurkCorpus/ASSET test outputs plus
  baseline outputs (ACCESS, MUSS, DMASS-DCSS, Dress-Ls, yats).
- `TS_T5.ipynb`: Colab notebook.
- Pins: Python 3.7.5, transformers 4.6.1, pytorch-lightning 1.7.2,
  spaCy 3.4.1, easse at commit `8df369f`.
- No LICENSE file.
- **Uncommitted, corrupted diff** in `source/model.py`
  (`self𝐆𝐫𝐮𝐯𝐛𝐨𝐱 𝐌𝐚𝐭𝐞𝐫𝐢𝐚𝐥.model = ...`): looks like injected text, a syntax
  error. Revert it. The `source/helper.py` diff only moves a function; drop it
  too unless the refactor needs it.

### Model input format

`"simplify: " + control tokens + sentence`, where each token is
`<ABBR>_<ratio>` (abbreviation = initials of the feature class name):

| Token | Feature | 4-token default | 5-token default |
|---|---|---|---|
| `W` | word-count ratio | – | 1.05 |
| `C` | character-length ratio | 0.95 | 0.95 |
| `L` | Levenshtein similarity ratio | 0.75 | 0.75 |
| `WR` | word-rank (lexical complexity) ratio | 0.75 | 0.75 |
| `DTD` | dependency-tree-depth ratio | 0.75 | 0.75 |

The builder must confirm the exact string (token order, separators) from
`Preprocessor.encode_sentence` and put a literal example in the model cards.

### Assets on D8 (`/Volumes/D8`)

| What | Path |
|---|---|
| 4-token model (t5-base, HF format) | `workspace/TS_T5/experiments/exp_1622148499982082_4tokens/pytorch_model.bin`, `config.json`, `params.json`, outputs, logs |
| 5-token model (Lightning ckpt only) | `workspace/TS_backup/TS_T5.2/experiments/exp_1623415175800762_5tokens/checkpoint-epoch=4-val_loss=0.83.ckpt` |
| Early 4-token run (max_len 256) | `workspace/TS_T5/experiments/exp_1614936426182236_C_WR_L_DTD/` |
| Exploratory / ablation runs | `workspace/TS_T5/results/exp_*`, `workspace/TS_backup/TS_T5.2/experiments/old/` |
| Ablation scores | `workspace/TS_T5/results/for_ablations.txt` (SARI/BLEU/FKGL per output-file hash) |
| Plots and scripts | `workspace/TS_T5/plots/` (compression ratio, Levenshtein), `notebooks/` |
| Human evaluation (381 MB) | `workspace/TS_T5/TS_T5 Human evaluation/` (`HIT_designs/`, `mTurk/`, `results/`, `report generation/`, `TS_T5 Human Evaluation.md`) |
| Later D8 git history | `workspace/TS_T5/.git` (commits like "add multi-model support", "plot compression and levenshtein distance") |

D8 is the only copy of some of this. Read-only: copy, never move or modify.

### Expected scores (README; the reproduction targets)

| Model | Tokens | TurkCorpus SARI / BLEU / FKGL | ASSET SARI / BLEU / FKGL |
|---|---|---|---|
| 4-token | C 0.95, L 0.75, WR 0.75, DTD 0.75 | 43.50 / 67.26 / 6.17 | 45.05 / 72.24 / 6.33 |
| 5-token | W 1.05, C 0.95, L 0.75, WR 0.75, DTD 0.75 | 43.34 / 65.54 / 5.55 | 45.26 / 72.19 / 5.92 |

## Approaches considered

**Code modernisation** (author picked A):

- **A. Modern dependencies, same results (picked).** Port to current
  toolchain, gate on reproducing the scores. Installs today; costs a careful
  port of preprocessing and Lightning code.
- B. Keep 2021 pins, clean up structure only. Exact original environment, but
  Python 3.7 and transformers 4.6.1 are hard to install now; fails the reuse
  goal.
- C. Modern inference-only package, legacy training folder. Less work, but
  training is no longer reproducible on the supported stack.

**Where results live** (author picked A):

- **A. GitHub `docs/` + HF model cards (picked).** One source of truth next to
  the code; cards summarise and link back.
- B. Plus an HF dataset repo with all outputs and logs. More discoverable, two
  places to keep in sync.
- C. One long report page. Reads well, harder to link individual results.

**Models to publish** (author picked 4-token + 5-token; rejected adding
t5-small or 4-token only).

**Human evaluation** (author picked anonymised + summarised; rejected
summary-only and leaving it out).

## Decisions made by the author

| # | Decision | Choice |
|---|---|---|
| 1 | Release goal | Reuse and citation |
| 2 | Models | 4-token + 5-token t5-base |
| 3 | Code | Modern deps, `pyproject.toml`, CLI; scores must match |
| 4 | Results | `docs/` on GitHub, HF cards link back |
| 5 | Human eval | Protocol + aggregates + per-item ratings, worker IDs replaced by salted hashes; raw exports stay private |
| 6 | Licence | CC BY-NC 4.0 for code and models (ACCESS and ASSET are CC BY-NC 4.0) |
| 7 | HF names | `kcsheang/ts-t5-base-4tokens`, `kcsheang/ts-t5-base-5tokens`, plus a Collection |
| 8 | Corrupted `model.py` diff | Revert |
| 9 | Raw mTurk exports already in git (`Batch_v1-3.csv`, with `WorkerId`) | Leave as is; no history rewrite; anonymised CSVs added beside them |
| 10 | Training verification on the modern stack | Smoke test only, no full retrain |

## Design

### Target repository layout

```
pyproject.toml            # package `ts_t5`, CLI entry point `ts-t5`
LICENSE                   # CC BY-NC 4.0
README.md                 # what, quickstart (HF), results table, citation
CITATION.cff
src/ts_t5/
  features.py             # control-token features + Preprocessor (from preprocessor.py)
  data.py                 # datasets, download, paths (from resources/download_data/helper)
  model.py                # LightningModule + datasets
  train.py
  evaluate.py             # EASSE scoring, no module globals
  simplify.py             # load model (HF id or local dir), generate
  search.py               # Optuna control-token search (from find_best_tokens/hyperparameters_tuning)
  cli.py                  # `ts-t5 train|evaluate|simplify|search`
configs/                  # train config for each paper model (from params.json)
scripts/
  convert_ckpt_to_hf.py   # Lightning ckpt -> HF save_pretrained
  push_to_hub.py          # upload model + card
  anonymise_human_eval.py
docs/
  RESULTS.md              # main results vs baselines
  ABLATIONS.md            # per-token ablations, token-value sweeps
  FINDINGS.md             # what was learned, negative results included
  HUMAN_EVALUATION.md
  EXPERIMENTS.md          # index of runs: id, config, data, outcome
  REPRODUCE.md            # exact commands, hardware, run times
  figures/
model_cards/              # README.md for each HF repo
system_outputs/           # kept: paper outputs + baselines
human_evaluation/         # anonymised, small files only
tests/                    # smoke tests
TS_T5.ipynb               # updated to load from HF
```

`resources/` stays as tracked today (datasets, `DUMPS/glove.42B.300d.pk`
~42 MB, preprocessor pickle): a clean clone must evaluate with no extra
downloads. *(Builder's call: no stable upstream host exists for WikiLarge /
Wiki-Auto, and the GloVe pickle is slow to rebuild.)* Untracked caches stay
ignored.

### Behaviour that must not change

- Control-token string format and value rounding.
- Decoding: beam search, `num_beams=8`, `max_length` = the model's
  `max_seq_length` from `params.json`, same post-filter that drops outputs
  equal to the input.
- Evaluation: EASSE SARI/BLEU/FKGL on the TurkCorpus and ASSET test sets,
  same tokenisation and lowercasing settings as the pinned easse commit. If
  current easse changes scores, pin the old commit for evaluation and record
  the reason in `REPRODUCE.md`.

### Reproduction gate (before any upload)

0. Pick the weights. Download the current Google Drive zips
   (`download_data.py` ids) and hash them against the D8 copies. Same: use
   D8. Different: publish whichever copy reproduces the README scores; if
   both do, the Drive copy (what users have used and cited). For the 5-token
   run with two `val_loss=0.83` checkpoints, same rule; default `epoch=4`.
   *(Builder's call.)*
1. Convert the 5-token checkpoint to HF format; load both models with
   `T5ForConditionalGeneration.from_pretrained`.
2. Run `ts-t5 evaluate` on both test sets with the default tokens above.
3. Pass: every SARI within ±0.1 of the table, BLEU and FKGL within ±0.5.
   Also diff the regenerated outputs against `system_outputs/TS_T5_*tokens/`
   and report the share of identical lines.
   Target is the README table (it was produced by these models). Check it
   against the paper's tables; any difference is reported to the author and
   noted in `RESULTS.md`, not silently resolved. *(Builder's call.)*
4. Fail: stop and report to the author with the numbers. Do not upload.

### Hugging Face model cards

Each card: description and paper link, intended use and limits (English,
sentence-level, Wikipedia-style text, non-commercial), training data
(WikiLarge; confirm from `params.json`/logs), input format with a literal
example, what each token controls and recommended values, a `transformers`
snippet, the evaluation table with the exact eval settings, hyperparameters
from `params.json`, BibTeX, licence `cc-by-nc-4.0`, `base_model: t5-base`,
tags (`text-simplification`, `controllable-generation`). Upload weights as
`model.safetensors` with the tokenizer files. Create the HF Collection with
both models and the paper.

`download_data.py`'s Google Drive downloads are replaced by HF Hub loading;
the README notes the old links are retired.

### Results and findings docs

- `RESULTS.md`: main table (both models + baselines from `system_outputs/`,
  scored by the same eval code), example simplifications.
- `ABLATIONS.md`: per-token ablation and token-value sweeps rebuilt from
  `for_ablations.txt` and run outputs. The scores there are keyed by output
  file hash; recover the token values for each hash from the run logs or by
  re-hashing token settings. Rows that cannot be traced are dropped, not
  guessed.
- `FINDINGS.md`: what each token does to outputs, the effect of the 5th token,
  the token search results, failure cases, negative results.
- `EXPERIMENTS.md`: one row per D8 run that is still readable.
- Figures from `plots/` regenerated as PNG/SVG into `docs/figures/`.

Every number in these docs must come from a file on D8 or a fresh run, with
its source noted. Nothing is filled in from memory of the paper.

### Human evaluation

The existing raw exports in the repo are left untouched (decision 9).
`scripts/anonymise_human_eval.py` reads the D8 mTurk results, replaces worker
IDs (and assignment/HIT IDs) with salted SHA-256 prefixes, drops free-text
and any column that can carry personal data, and writes small CSVs into
`human_evaluation/`. The salt stays local, uncommitted. Before commit, grep
the output for the original ID pattern (`A[0-9A-Z]{12,}`) and fail if any
match. HIT designs and instructions are published as-is after a manual
check for personal data.

### Training verification

No full retrain. A smoke run (~100 steps on a WikiLarge subset, CPU or MPS)
must show falling loss and a checkpoint that saves, converts to HF format and
loads. `REPRODUCE.md` lists the original hardware, settings and run times
taken from the D8 logs and `params.json`, and says plainly that the full
training was not re-run on the new stack.

### Tests

`tests/` smoke tests: preprocessor produces the exact control-token string
for a fixed sentence; `simplify` on 3 sentences with the 4-token model
returns non-empty, non-identical outputs; anonymiser leaves no worker ID.
CI: GitHub Actions on CPU running the smoke tests (model test marked slow,
optional).

## Release sequence and confirmations

All local work first. Each outward step needs the author's explicit OK:

1. Commit the refactor and docs on a branch; push the branch (ask).
2. Create the HF repos **private**, upload, author reviews the cards (ask).
3. Make the HF repos public and create the Collection (ask).
4. Merge to `main` and push (ask).

## Left to the builder

- Exact current versions to pin (latest stable that runs the reproduction).
- Internal module boundaries beyond the layout above.
- Which exploratory runs are worth an `EXPERIMENTS.md` row.
- Figure styling.
- Whether the Colab notebook is kept as one notebook or split.
