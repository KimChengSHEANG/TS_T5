# Ablations

Control-token ablations from the 2020-21 experiments. All numbers come from one file:
`/Volumes/D8/workspace/TS_T5/results/for_ablations.txt` (D8 backup, read-only; not in this repo).
Each score was made by the original pipeline (easse of that time), on the **test** sets (`turkcorpus_test`, `asset_test`).
Most rows were not re-scored. The headline rows and the rows that contradict the paper were, and the pinned easse gives
the same values to within 0.01. They are other decoding passes than the README table in `docs/RESULTS.md`.
Values ending in zero were fed as `0.8` or `1.0`, a format absent from training (`docs/FINDINGS.md`, finding 5).

## Method: from output-file hash back to token values

The file lists one output file per row, named `<hash>_<test set>.test.txt`, with its SARI, BLEU and FKGL.
The hash is `generate_hash(features_kwargs)` (md5 of `str(dict)`, in `ts_t5.data`). The token values are not in the file.
We recovered them with `python scripts/make_docs_tables.py ablations`:

1. Read every `score_*.txt` log under the D8 folders. Each log line prints token values, for example `C: 0.95 L: 0.75 ...`.
   Build a `features_kwargs` dict from the line. Accept it only if `generate_hash` gives the hash in the log's file name.
   This gave the token set and the token order for most hashes. Values such as `0.30000000000000004` are kept exactly,
   because they change the hash.
2. For hashes with no log, brute-force `generate_hash` over every ordered subset of the 5 tokens
   (`W, C, L, WR, DTD`): all values of the grid seen in the logs for 1 to 3 tokens, and an 11-value grid
   (0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15) for 4 and 5 tokens.
3. The empty dict (hash `99914b932bd37a50b983c5e7c90ae93b`) is the no-token model.
4. Sanity check: the 4-token default maps to `d859b9c4aab9f7a9c719167728399278` and the 5-token default to
   `267a9c5912b4c0e37bf8bf7260f7576d`. Both appear in the table below.

Result: **241 rows, 121 distinct hashes. 241 rows matched. 0 rows dropped.**
(Footnote: the plan expected some unmatched rows. There were none. Every hash is reproduced by exactly the token
dict printed in the table.)

Token order is part of the hash, and the table lists tokens in the order that matches the hash.
The run names (for example `W_C_WR_L`) describe which tokens the model was trained with, not this order.

## Which models these are

Run names and settings are in `docs/EXPERIMENTS.md`. The two that matter most:

- `exp_1607800772213084_C_WR_L_DTD` (Dec 2020) is the released 4-token model. Its `pytorch_model.bin` has sha256
  `e693d5bd...`, the same as the Drive zip.
- `exp_1601582154282931_best` (Sep 2020) is the released 5-token model, sha256 `36091a14...`, same as the Drive zip.
  Its 60 rows (30 per test set) cover token subsets and values.
- The other `exp_16114...`-`exp_16118...` runs (Jan 2021) were trained with a single token or a small subset, t5-base,
  `max_seq_length` 256, batch 6 x 16. Single-token runs: `_C`, `_W`, `_L`, `_WR`, `_DTD`.
- `exp_1601753330122698_base_no_tokens` and `exp_1602001102227427_small_no_tokens` have no tokens at all.
- `exp_1601752818273947_small` is a t5-small model with 5 tokens.

## What each token does alone

Models trained with one token (target ratio 0.8 at training time), scored on test with that token at 11 values.
Source: the `exp_1611667703272792_C`, `_W`, `_L`, `_WR` and `_DTD` sections below.
Baseline with no tokens at all (t5-base, `exp_1601753330122698_base_no_tokens`): TurkCorpus SARI 37.56, ASSET 34.15.

| Token | TurkCorpus best SARI (value) | TurkCorpus at 0.75 | TurkCorpus worst (value) | ASSET best SARI (value) | ASSET at 0.75 | ASSET worst (value) |
|---|---|---|---|---|---|---|
| C | 38.95 (0.75) | 38.95 | 25.06 (0.25) | 39.59 (0.50) | 39.30 | 26.53 (1.10) |
| W | 38.86 (0.85) | 37.96 | 24.19 (0.25) | 38.52 (0.75) | 38.52 | 28.48 (1.10) |
| L | 40.91 (0.85) | 39.23 | 21.21 (0.25) | 41.59 (1.05) | 41.50 | 28.32 (0.25) |
| WR | 40.45 (0.85) | 39.77 | 33.31 (0.25) | 41.41 (0.25) | 40.76 | 32.85 (1.00) |
| DTD | 39.18 (0.75) | 39.18 | 22.88 (0.25) | 40.08 (0.50) | 37.82 | 30.55 (0.25) |

SARI by token value (TurkCorpus test, then ASSET test):

| Token | 0.25 | 0.50 | 0.75 | 0.85 | 0.95 | 1.00 | 1.05 | 1.10 | 1.15 |
|---|---|---|---|---|---|---|---|---|---|
| C (Turk) | 25.1 | 36.6 | 39.0 | 38.7 | 35.7 | 31.8 | 32.9 | 32.0 | 35.7 |
| W (Turk) | 24.2 | 36.2 | 38.0 | 38.9 | 35.0 | 32.9 | 34.3 | 34.9 | 37.6 |
| L (Turk) | 21.2 | 35.1 | 39.2 | 40.9 | 36.4 | 35.2 | 39.1 | 36.6 | 38.4 |
| WR (Turk) | 33.3 | 33.4 | 39.8 | 40.5 | 38.4 | 35.7 | 36.0 | 37.0 | 37.2 |
| DTD (Turk) | 22.9 | 37.7 | 39.2 | 38.3 | 36.4 | 35.0 | 34.7 | 35.3 | 35.2 |
| C (ASSET) | 32.0 | 39.6 | 39.3 | 37.3 | 32.2 | 27.6 | 27.7 | 26.5 | 30.7 |
| W (ASSET) | 31.2 | 38.3 | 38.5 | 37.5 | 33.0 | 28.9 | 28.7 | 28.5 | 31.9 |
| L (ASSET) | 28.3 | 40.2 | 41.5 | 40.1 | 32.3 | 38.0 | 41.6 | 39.2 | 40.7 |
| WR (ASSET) | 41.4 | 41.1 | 40.8 | 38.5 | 35.1 | 32.9 | 33.7 | 35.3 | 36.1 |
| DTD (ASSET) | 30.6 | 40.1 | 37.8 | 35.3 | 32.2 | 30.9 | 30.7 | 31.1 | 31.1 |

Reading the table:

- At its best value, each single token adds 1.3 to 3.4 SARI points over no tokens on TurkCorpus
  (37.56 to 38.86-40.91). On ASSET the gain is 4.4 to 7.4 points (34.15 to 38.52-41.59).
- L and WR help most on TurkCorpus (+3.4 and +2.9). C, W and DTD help less (+1.3 to +1.6).
- The best value is not the same for each test set. ASSET prefers more compression for C, DTD and WR
  (0.5, 0.5 and 0.25). TurkCorpus prefers values near 0.75 to 0.85 for all five tokens.
- On TurkCorpus the lowest SARI is at 0.25. At 0.95 to 1.15, SARI is 2 to 7 points below the best value.
- In most rows with a value of 0.9 or more, BLEU is above 85: the model nearly copies the input
  (see the full table). The L model at 1.0 to 1.15 is an exception (BLEU 57 to 76 on TurkCorpus).
- Values at 0.25 give very low BLEU (for example L 0.25: BLEU 3.43 on TurkCorpus). The model cuts too much.
- The L curve on ASSET is not smooth (32.3 at 0.95, 38.0 at 1.0, 41.6 at 1.05). We have one run per point
  and no repeats, so differences of 1 to 2 SARI points are within what we cannot separate.

## Adding tokens at a shared value

Models trained with a subset of tokens, test scores with all their tokens at the same value
(C 0.95 and W 1.05 where those tokens are present). SARI / BLEU / FKGL, from the sections named in the first column.

| Model (run) | Tokens at inference | TurkCorpus | ASSET |
|---|---|---|---|
| `exp_1611533249297539_WR_L` | WR 0.75, L 0.75 | 40.39 / 74.35 / 6.20 | 42.85 / 80.39 / 6.24 |
| `exp_1611575591742851_C_WR_L` | C 0.95, WR 0.75, L 0.75 | 42.91 / 67.10 / 6.53 | 44.63 / 71.60 / 6.59 |
| `exp_1611620290763589_W_C_WR_L` | W 1.05, C 0.95, L 0.75, WR 0.75 | 43.03 / 68.11 / 5.97 | 44.11 / 73.85 / 6.24 |
| `exp_1607800772213084_C_WR_L_DTD` (4-token, released) | C 0.95, L 0.75, WR 0.75, DTD 0.75 | 43.31 / 66.23 / 6.17 | 44.91 / 71.96 / 6.32 |
| `exp_1601582154282931_best` (5-token, released) | W 1.05, C 0.95, L 0.75, WR 0.75, DTD 0.75 | 43.00 / 64.42 / 5.63 | 45.04 / 71.21 / 5.88 |

- Going from 2 tokens to 3 (adding C) is the biggest step: +2.5 SARI on TurkCorpus, +1.8 on ASSET.
- The later steps are small and have mixed sign. TurkCorpus: +0.12 (add W), +0.28 (add DTD), -0.31 (add W to the
  4-token model). ASSET: -0.52, +0.80, +0.13. Each row is a different training run, with one run per row.
  We did not measure run-to-run noise (see `docs/FINDINGS.md`, "What we could not test"), so we do not read
  these steps as real effects.
- Higher token values at the same set are worse: with all tokens at 0.85 the released 4-token model gets 41.07 (Turk,
  token order WR, L, DTD; the table row `C0.95 WR0.85 L0.85 DTD0.85`) and 42.67 (order L, WR, DTD).

## Token order matters at inference

The released 4-token model gives different scores when the same values are given in another order:

| Order given | TurkCorpus SARI / BLEU / FKGL | ASSET SARI / BLEU / FKGL |
|---|---|---|
| C, L, WR, DTD (default, hash `d859b9c4`) | 43.31 / 66.23 / 6.17 | 44.91 / 71.96 / 6.32 |
| C, WR, L, DTD (all 0.75) | 42.61 / 77.89 / 6.63 | 41.90 / 80.39 / 6.82 |

Source: section `exp_1607800772213084_C_WR_L_DTD` below. The order must match the order used in training.
The code keeps the canonical order `W, C, L, WR, DTD` (`ts_t5.features.TOKEN_CLASSES`).

## Dropping tokens from the 5-token model at inference

The released 5-token model, scored with only some tokens in the input (section `exp_1601582154282931_best`).
The full 5 tokens at 0.75 give 43.00 (Turk) and 45.04 (ASSET).

- Best subset on TurkCorpus: `C 1.0, L 0.5` with 41.92 / 50.89 / 7.22. Best on ASSET: `C 0.25` with 39.29 / 73.96 / 5.94.
- Every subset we tried is below the full set (1.08 or more SARI lower on TurkCorpus, 5.75 or more on ASSET).
- Subsets with `L 0.25`, `W 1.0` or `L 1.0` are among the lowest (25.1 to 30.8 SARI).

## Full table

Every matched row. `Hash` is the first 8 characters of the output-file hash. Test set `turkcorpus` or `asset`.
Sorted by test set, then by token order and value.

### `exp_1601582154282931_best`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | C=0.25 | 39.29 | 73.96 | 5.94 | `1a3e437e` |
| asset | C=0.50 | 34.85 | 70.76 | 8.06 | `a0acd266` |
| asset | C=0.75 | 34.24 | 63.85 | 8.38 | `fee07a82` |
| asset | C=0.85 | 31.42 | 60.74 | 8.89 | `ac8d6de8` |
| asset | C=0.95 | 27.65 | 59.23 | 10.40 | `ccd77d8e` |
| asset | C=1.00 | 28.26 | 49.84 | 9.69 | `40078f40` |
| asset | C=1.00 DTD=0.25 | 31.28 | 64.56 | 7.28 | `f44bf268` |
| asset | C=1.00 DTD=0.50 | 28.55 | 58.46 | 8.34 | `b9c218c4` |
| asset | C=1.00 DTD=0.75 | 28.21 | 59.10 | 8.47 | `33daf002` |
| asset | C=1.00 DTD=1.00 | 26.28 | 54.67 | 10.96 | `d42cd404` |
| asset | C=1.00 L=0.25 | 33.43 | 9.26 | 6.18 | `02df34fb` |
| asset | C=1.00 L=0.50 | 39.26 | 55.15 | 7.67 | `17a909e5` |
| asset | C=1.00 L=0.75 | 37.54 | 60.12 | 8.17 | `36f6b25c` |
| asset | C=1.00 L=1.00 | 25.41 | 62.54 | 10.19 | `e7e3e4cd` |
| asset | C=1.00 W=0.25 | 38.10 | 32.19 | 5.29 | `422d40a3` |
| asset | C=1.00 W=0.50 | 35.99 | 62.82 | 7.68 | `c81082bc` |
| asset | C=1.00 W=0.75 | 36.33 | 60.19 | 8.01 | `23278865` |
| asset | C=1.00 W=1.00 | 25.08 | 55.46 | 10.73 | `39923bc5` |
| asset | C=1.00 WR=0.25 | 35.32 | 63.23 | 7.44 | `fc667703` |
| asset | C=1.00 WR=0.50 | 33.07 | 62.56 | 8.10 | `06d4275e` |
| asset | C=1.00 WR=0.75 | 34.27 | 63.79 | 8.08 | `b56f7228` |
| asset | C=1.00 WR=1.00 | 26.95 | 64.48 | 9.84 | `9adc9865` |
| asset | W=0.25 | 38.57 | 69.19 | 6.43 | `84a96e5b` |
| asset | W=0.50 | 34.08 | 72.43 | 8.78 | `5ea354b6` |
| asset | W=0.75 | 31.76 | 63.05 | 9.11 | `0064aa19` |
| asset | W=0.85 | 29.87 | 58.81 | 9.41 | `c974e305` |
| asset | W=0.95 | 26.63 | 55.57 | 10.61 | `c8deffac` |
| asset | W=1.00 | 27.48 | 44.98 | 10.10 | `a829cd94` |
| asset | W=1.05 C=0.95 L=0.75 WR=0.75 DTD=0.75 | 45.04 | 71.21 | 5.88 | `267a9c59` |
| asset | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 42.17 | 80.08 | 6.77 | `dbd23931` |
| turkcorpus | C=0.25 | 35.20 | 55.77 | 5.34 | `1a3e437e` |
| turkcorpus | C=0.50 | 38.66 | 65.50 | 7.59 | `a0acd266` |
| turkcorpus | C=0.75 | 38.50 | 60.30 | 7.98 | `fee07a82` |
| turkcorpus | C=0.85 | 36.53 | 58.86 | 8.66 | `ac8d6de8` |
| turkcorpus | C=0.95 | 32.77 | 58.25 | 9.66 | `ccd77d8e` |
| turkcorpus | C=1.00 | 33.46 | 49.68 | 9.59 | `40078f40` |
| turkcorpus | C=1.00 DTD=0.25 | 34.66 | 60.49 | 7.29 | `f44bf268` |
| turkcorpus | C=1.00 DTD=0.50 | 33.05 | 56.75 | 8.30 | `b9c218c4` |
| turkcorpus | C=1.00 DTD=0.75 | 32.95 | 57.27 | 8.50 | `33daf002` |
| turkcorpus | C=1.00 DTD=1.00 | 31.56 | 54.18 | 10.46 | `d42cd404` |
| turkcorpus | C=1.00 L=0.25 | 25.72 | 6.63 | 6.28 | `02df34fb` |
| turkcorpus | C=1.00 L=0.50 | 41.92 | 50.89 | 7.22 | `17a909e5` |
| turkcorpus | C=1.00 L=0.75 | 40.36 | 56.92 | 7.90 | `36f6b25c` |
| turkcorpus | C=1.00 L=1.00 | 30.51 | 59.96 | 9.81 | `e7e3e4cd` |
| turkcorpus | C=1.00 W=0.25 | 30.80 | 21.11 | 4.98 | `422d40a3` |
| turkcorpus | C=1.00 W=0.50 | 39.76 | 58.22 | 7.54 | `c81082bc` |
| turkcorpus | C=1.00 W=0.75 | 40.46 | 57.25 | 7.75 | `23278865` |
| turkcorpus | C=1.00 W=1.00 | 29.90 | 54.94 | 10.31 | `39923bc5` |
| turkcorpus | C=1.00 WR=0.25 | 38.91 | 57.63 | 7.23 | `fc667703` |
| turkcorpus | C=1.00 WR=0.50 | 37.38 | 59.82 | 7.96 | `06d4275e` |
| turkcorpus | C=1.00 WR=0.75 | 38.42 | 60.85 | 7.99 | `b56f7228` |
| turkcorpus | C=1.00 WR=1.00 | 32.01 | 62.27 | 9.39 | `9adc9865` |
| turkcorpus | W=0.25 | 34.05 | 52.68 | 6.06 | `84a96e5b` |
| turkcorpus | W=0.50 | 37.40 | 67.76 | 8.44 | `5ea354b6` |
| turkcorpus | W=0.75 | 36.60 | 61.24 | 8.66 | `0064aa19` |
| turkcorpus | W=0.85 | 34.90 | 58.29 | 9.15 | `c974e305` |
| turkcorpus | W=0.95 | 31.16 | 55.25 | 10.07 | `c8deffac` |
| turkcorpus | W=1.00 | 33.20 | 44.00 | 9.60 | `a829cd94` |
| turkcorpus | W=1.05 C=0.95 L=0.75 WR=0.75 DTD=0.75 | 43.00 | 64.42 | 5.63 | `267a9c59` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 42.98 | 75.15 | 6.58 | `dbd23931` |

### `exp_1601752818273947_small`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | W=1.05 C=0.95 L=0.75 WR=0.75 DTD=0.75 | 39.13 | 86.08 | 7.00 | `267a9c59` |
| asset | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 35.16 | 89.09 | 7.86 | `dbd23931` |
| turkcorpus | W=1.00 C=0.90 L=0.80 WR=0.80 DTD=0.80 | 37.26 | 90.01 | 9.03 | `7c2d8298` |
| turkcorpus | W=1.00 C=0.90 L=0.80 WR=0.80 DTD=0.85 | 38.65 | 87.73 | 7.58 | `4b2c7efd` |
| turkcorpus | W=1.00 C=0.90 L=0.80 WR=0.85 DTD=0.80 | 37.46 | 90.17 | 9.01 | `c6bb870c` |
| turkcorpus | W=1.00 C=0.90 L=0.80 WR=0.85 DTD=0.85 | 38.93 | 87.45 | 7.53 | `7e3502db` |
| turkcorpus | W=1.00 C=0.90 L=0.85 WR=0.80 DTD=0.80 | 37.26 | 90.17 | 9.04 | `a5fec080` |
| turkcorpus | W=1.00 C=0.90 L=0.85 WR=0.80 DTD=0.85 | 38.81 | 87.60 | 7.51 | `d80cc858` |
| turkcorpus | W=1.00 C=0.90 L=0.85 WR=0.85 DTD=0.80 | 37.57 | 89.95 | 9.03 | `dc6fb732` |
| turkcorpus | W=1.00 C=0.90 L=0.85 WR=0.85 DTD=0.85 | 39.07 | 87.15 | 7.48 | `09ae0c65` |
| turkcorpus | W=1.00 C=0.95 L=0.80 WR=0.80 DTD=0.80 | 37.39 | 90.15 | 9.02 | `498b1d78` |
| turkcorpus | W=1.00 C=0.95 L=0.80 WR=0.80 DTD=0.85 | 38.82 | 87.43 | 7.49 | `7c677a52` |
| turkcorpus | W=1.00 C=0.95 L=0.80 WR=0.85 DTD=0.80 | 37.56 | 89.93 | 9.01 | `25c22c40` |
| turkcorpus | W=1.00 C=0.95 L=0.80 WR=0.85 DTD=0.85 | 39.07 | 87.23 | 7.53 | `a1df7c04` |
| turkcorpus | W=1.00 C=0.95 L=0.85 WR=0.80 DTD=0.80 | 37.47 | 90.11 | 9.04 | `a6eace55` |
| turkcorpus | W=1.00 C=0.95 L=0.85 WR=0.80 DTD=0.85 | 38.98 | 87.24 | 7.51 | `8163a8c0` |
| turkcorpus | W=1.00 C=0.95 L=0.85 WR=0.85 DTD=0.80 | 37.73 | 89.61 | 8.95 | `d015093a` |
| turkcorpus | W=1.00 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 39.23 | 87.46 | 7.53 | `0422eec3` |
| turkcorpus | W=1.05 C=0.90 L=0.80 WR=0.80 DTD=0.80 | 37.40 | 90.10 | 9.02 | `41f850aa` |
| turkcorpus | W=1.05 C=0.90 L=0.80 WR=0.80 DTD=0.85 | 38.76 | 87.64 | 7.46 | `65e08373` |
| turkcorpus | W=1.05 C=0.90 L=0.80 WR=0.85 DTD=0.80 | 37.52 | 90.09 | 9.01 | `161fa7cb` |
| turkcorpus | W=1.05 C=0.90 L=0.80 WR=0.85 DTD=0.85 | 39.04 | 87.31 | 7.47 | `39f5075a` |
| turkcorpus | W=1.05 C=0.90 L=0.85 WR=0.80 DTD=0.80 | 37.43 | 90.17 | 9.01 | `3c8a241e` |
| turkcorpus | W=1.05 C=0.90 L=0.85 WR=0.80 DTD=0.85 | 38.87 | 87.23 | 7.43 | `68d193ff` |
| turkcorpus | W=1.05 C=0.90 L=0.85 WR=0.85 DTD=0.80 | 37.63 | 89.81 | 9.02 | `39f891fd` |
| turkcorpus | W=1.05 C=0.90 L=0.85 WR=0.85 DTD=0.85 | 39.06 | 87.03 | 7.45 | `8f45c27c` |
| turkcorpus | W=1.05 C=0.95 L=0.75 WR=0.75 DTD=0.75 | 40.83 | 85.12 | 6.78 | `267a9c59` |
| turkcorpus | W=1.05 C=0.95 L=0.80 WR=0.80 DTD=0.80 | 37.37 | 90.25 | 9.04 | `23ca2069` |
| turkcorpus | W=1.05 C=0.95 L=0.80 WR=0.80 DTD=0.85 | 38.81 | 87.48 | 7.42 | `54db30ac` |
| turkcorpus | W=1.05 C=0.95 L=0.80 WR=0.85 DTD=0.80 | 37.58 | 90.17 | 9.04 | `20549c58` |
| turkcorpus | W=1.05 C=0.95 L=0.80 WR=0.85 DTD=0.85 | 39.00 | 87.30 | 7.46 | `bea0e73e` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.80 DTD=0.80 | 37.47 | 90.03 | 8.99 | `464aae58` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.80 DTD=0.85 | 38.98 | 87.15 | 7.42 | `660b75c0` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.80 | 37.73 | 89.71 | 9.01 | `dfc13668` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 38.52 | 90.17 | 7.73 | `dbd23931` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.85 DTD=0.85 | 39.23 | 87.32 | 7.45 | `dbd23931` |

### `exp_1601753330122698_base_no_tokens`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | (no tokens) | 34.15 | 88.97 | 8.95 | `99914b93` |
| turkcorpus | (no tokens) | 37.56 | 90.96 | 8.81 | `99914b93` |

### `exp_1602001102227427_small_no_tokens`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | (no tokens) | 29.85 | 90.39 | 9.49 | `99914b93` |
| turkcorpus | (no tokens) | 34.50 | 94.17 | 9.44 | `99914b93` |

### `exp_1607800772213084_C_WR_L_DTD`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | C=0.95 L=0.75 WR=0.75 DTD=0.75 | 44.91 | 71.96 | 6.32 | `d859b9c4` |
| asset | C=0.95 WR=0.75 L=0.75 DTD=0.75 | 41.90 | 80.39 | 6.82 | `1c3ed59a` |
| asset | C=0.95 WR=0.85 L=0.85 DTD=0.85 | 38.88 | 85.74 | 7.86 | `e3ea6a29` |
| turkcorpus | C=0.95 L=0.75 WR=0.75 DTD=0.75 | 43.31 | 66.23 | 6.17 | `d859b9c4` |
| turkcorpus | C=0.95 L=0.85 WR=0.85 DTD=0.85 | 42.67 | 78.48 | 7.04 | `d68f71c5` |
| turkcorpus | C=0.95 WR=0.75 L=0.75 DTD=0.75 | 42.61 | 77.89 | 6.63 | `1c3ed59a` |
| turkcorpus | C=0.95 WR=0.85 L=0.85 DTD=0.85 | 41.07 | 84.76 | 7.81 | `e3ea6a29` |

### `exp_1611490086932860_DTD`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | DTD=0.25 | 30.55 | 15.31 | 3.51 | `1dc332eb` |
| asset | DTD=0.50 | 40.08 | 81.94 | 6.56 | `ff0ca00a` |
| asset | DTD=0.75 | 37.82 | 87.25 | 7.92 | `8018cedc` |
| asset | DTD=0.80 | 34.71 | 88.78 | 8.73 | `7707025e` |
| asset | DTD=0.85 | 35.34 | 88.88 | 8.48 | `ad705e07` |
| asset | DTD=0.90 | 30.69 | 89.12 | 9.37 | `6aa18a29` |
| asset | DTD=0.95 | 32.18 | 89.29 | 9.12 | `b3f36ee4` |
| asset | DTD=1.00 | 30.87 | 88.92 | 9.49 | `d28a15f7` |
| asset | DTD=1.05 | 30.71 | 89.24 | 9.43 | `fd71417c` |
| asset | DTD=1.10 | 31.05 | 88.58 | 9.43 | `c8324654` |
| asset | DTD=1.15 | 31.09 | 89.17 | 9.36 | `e59be7e0` |
| turkcorpus | DTD=0.25 | 22.88 | 5.46 | 3.68 | `1dc332eb` |
| turkcorpus | DTD=0.50 | 37.66 | 71.64 | 6.31 | `ff0ca00a` |
| turkcorpus | DTD=0.75 | 39.18 | 87.60 | 7.82 | `8018cedc` |
| turkcorpus | DTD=0.80 | 38.09 | 90.85 | 8.42 | `7707025e` |
| turkcorpus | DTD=0.85 | 38.26 | 91.23 | 8.51 | `ad705e07` |
| turkcorpus | DTD=0.90 | 35.30 | 94.18 | 9.41 | `6aa18a29` |
| turkcorpus | DTD=0.95 | 36.38 | 94.04 | 9.15 | `b3f36ee4` |
| turkcorpus | DTD=1.00 | 35.00 | 93.92 | 9.46 | `d28a15f7` |
| turkcorpus | DTD=1.05 | 34.70 | 94.50 | 9.48 | `fd71417c` |
| turkcorpus | DTD=1.10 | 35.28 | 93.57 | 9.47 | `c8324654` |
| turkcorpus | DTD=1.15 | 35.24 | 94.35 | 9.45 | `e59be7e0` |

### `exp_1611533249297539_WR_L`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | WR=0.50 L=0.50 | 40.96 | 63.04 | 4.47 | `c519f4f8` |
| asset | WR=0.75 L=0.75 | 42.85 | 80.39 | 6.24 | `59c06e45` |
| asset | WR=0.85 L=0.85 | 41.44 | 84.66 | 7.33 | `41f08ff2` |
| asset | WR=1.00 L=1.00 | 36.90 | 85.57 | 8.29 | `7cb0d4f6` |
| turkcorpus | WR=0.50 L=0.50 | 35.17 | 47.74 | 4.34 | `c519f4f8` |
| turkcorpus | WR=0.75 L=0.75 | 40.39 | 74.35 | 6.20 | `59c06e45` |
| turkcorpus | WR=0.85 L=0.85 | 41.72 | 83.91 | 7.43 | `41f08ff2` |
| turkcorpus | WR=1.00 L=1.00 | 37.96 | 88.37 | 8.31 | `7cb0d4f6` |

### `exp_1611575591742851_C_WR_L`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | C=0.95 WR=0.50 L=0.50 | 44.90 | 56.76 | 5.93 | `e1c155c6` |
| asset | C=0.95 WR=0.75 L=0.75 | 44.63 | 71.60 | 6.59 | `a5814ba2` |
| asset | C=0.95 WR=0.85 L=0.85 | 41.79 | 81.88 | 7.47 | `7b4efdea` |
| asset | C=0.95 WR=1.00 L=1.00 | 28.68 | 88.78 | 9.78 | `9557973e` |
| turkcorpus | C=0.95 WR=0.50 L=0.50 | 40.60 | 49.74 | 5.85 | `e1c155c6` |
| turkcorpus | C=0.95 WR=0.75 L=0.75 | 42.91 | 67.10 | 6.53 | `a5814ba2` |
| turkcorpus | C=0.95 WR=0.85 L=0.85 | 42.62 | 79.45 | 7.56 | `7b4efdea` |
| turkcorpus | C=0.95 WR=1.00 L=1.00 | 31.62 | 95.37 | 9.78 | `9557973e` |

### `exp_1611620290763589_W_C_WR_L`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | W=1.05 C=0.95 L=0.50 WR=0.50 | 44.65 | 58.52 | 5.53 | `8380ea7f` |
| asset | W=1.05 C=0.95 L=0.75 WR=0.75 | 44.11 | 73.85 | 6.24 | `6aa569c6` |
| asset | W=1.05 C=0.95 L=0.85 WR=0.85 | 40.90 | 81.89 | 7.25 | `21a292f5` |
| asset | W=1.05 C=0.95 L=1.00 WR=1.00 | 30.02 | 86.10 | 9.00 | `10ce2c36` |
| turkcorpus | W=1.05 C=0.95 L=0.50 WR=0.50 | 40.88 | 51.17 | 5.22 | `8380ea7f` |
| turkcorpus | W=1.05 C=0.95 L=0.75 WR=0.75 | 43.03 | 68.11 | 5.97 | `6aa569c6` |
| turkcorpus | W=1.05 C=0.95 L=0.85 WR=0.85 | 42.50 | 78.65 | 7.17 | `21a292f5` |
| turkcorpus | W=1.05 C=0.95 L=1.00 WR=1.00 | 35.42 | 87.44 | 8.64 | `10ce2c36` |

### `exp_1611667703272792_C`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | C=0.25 | 32.05 | 23.42 | 2.25 | `1a3e437e` |
| asset | C=0.50 | 39.59 | 79.22 | 6.06 | `a0acd266` |
| asset | C=0.75 | 39.30 | 85.68 | 7.64 | `fee07a82` |
| asset | C=0.80 | 35.71 | 88.47 | 8.79 | `5dec98a8` |
| asset | C=0.85 | 37.33 | 87.37 | 8.50 | `ac8d6de8` |
| asset | C=0.90 | 30.72 | 89.10 | 9.60 | `a40a2631` |
| asset | C=0.95 | 32.21 | 89.11 | 9.41 | `ccd77d8e` |
| asset | C=1.00 | 27.63 | 89.32 | 9.73 | `40078f40` |
| asset | C=1.05 | 27.69 | 88.54 | 9.54 | `e5689e2d` |
| asset | C=1.10 | 26.53 | 88.63 | 9.76 | `67aa78c9` |
| asset | C=1.15 | 30.66 | 82.70 | 8.91 | `3e42ad50` |
| turkcorpus | C=0.25 | 25.06 | 12.05 | 2.01 | `1a3e437e` |
| turkcorpus | C=0.50 | 36.58 | 68.67 | 6.02 | `a0acd266` |
| turkcorpus | C=0.75 | 38.95 | 84.81 | 7.76 | `fee07a82` |
| turkcorpus | C=0.80 | 37.95 | 91.86 | 8.93 | `5dec98a8` |
| turkcorpus | C=0.85 | 38.71 | 90.07 | 8.64 | `ac8d6de8` |
| turkcorpus | C=0.90 | 34.46 | 94.47 | 9.63 | `a40a2631` |
| turkcorpus | C=0.95 | 35.71 | 93.95 | 9.48 | `ccd77d8e` |
| turkcorpus | C=1.00 | 31.76 | 95.37 | 9.88 | `40078f40` |
| turkcorpus | C=1.05 | 32.94 | 92.66 | 9.54 | `e5689e2d` |
| turkcorpus | C=1.10 | 32.01 | 93.08 | 9.83 | `67aa78c9` |
| turkcorpus | C=1.15 | 35.73 | 82.05 | 8.83 | `3e42ad50` |

### `exp_1611714803808738_W`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | W=0.25 | 31.20 | 19.60 | 3.15 | `84a96e5b` |
| asset | W=0.50 | 38.31 | 76.63 | 6.08 | `5ea354b6` |
| asset | W=0.75 | 38.52 | 84.03 | 7.46 | `0064aa19` |
| asset | W=0.80 | 35.84 | 87.79 | 8.79 | `837602d7` |
| asset | W=0.85 | 37.45 | 86.93 | 8.37 | `c974e305` |
| asset | W=0.90 | 31.29 | 88.85 | 9.47 | `e8029e0a` |
| asset | W=0.95 | 32.96 | 88.50 | 9.26 | `c8deffac` |
| asset | W=1.00 | 28.95 | 89.32 | 9.49 | `a829cd94` |
| asset | W=1.05 | 28.73 | 89.19 | 9.46 | `9552bc9d` |
| asset | W=1.10 | 28.48 | 88.67 | 9.54 | `57693155` |
| asset | W=1.15 | 31.92 | 84.94 | 8.63 | `06801a31` |
| turkcorpus | W=0.25 | 24.19 | 9.36 | 2.12 | `84a96e5b` |
| turkcorpus | W=0.50 | 36.19 | 66.73 | 6.14 | `5ea354b6` |
| turkcorpus | W=0.75 | 37.96 | 83.04 | 7.70 | `0064aa19` |
| turkcorpus | W=0.80 | 37.34 | 91.73 | 9.01 | `837602d7` |
| turkcorpus | W=0.85 | 38.86 | 89.11 | 8.62 | `c974e305` |
| turkcorpus | W=0.90 | 33.50 | 94.42 | 9.61 | `e8029e0a` |
| turkcorpus | W=0.95 | 35.05 | 93.60 | 9.51 | `c8deffac` |
| turkcorpus | W=1.00 | 32.85 | 94.08 | 9.72 | `a829cd94` |
| turkcorpus | W=1.05 | 34.30 | 92.36 | 9.42 | `9552bc9d` |
| turkcorpus | W=1.10 | 34.87 | 88.20 | 9.27 | `57693155` |
| turkcorpus | W=1.15 | 37.64 | 80.04 | 8.55 | `06801a31` |

### `exp_1611761596771104_L`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | L=0.25 | 28.32 | 8.40 | 1.53 | `f0afe9fd` |
| asset | L=0.50 | 40.21 | 65.83 | 5.60 | `5e26cf17` |
| asset | L=0.75 | 41.50 | 83.83 | 6.70 | `24d2fa8a` |
| asset | L=0.80 | 40.15 | 86.30 | 7.57 | `5e1090c1` |
| asset | L=0.85 | 40.13 | 86.15 | 7.56 | `aecec72c` |
| asset | L=0.90 | 32.01 | 89.54 | 9.23 | `df6d6b29` |
| asset | L=0.95 | 32.31 | 89.11 | 9.11 | `6805dee7` |
| asset | L=1.00 | 37.96 | 65.56 | 5.41 | `91fde837` |
| asset | L=1.05 | 41.59 | 82.52 | 6.54 | `9787cb78` |
| asset | L=1.10 | 39.25 | 71.16 | 5.91 | `655a7da7` |
| asset | L=1.15 | 40.74 | 80.38 | 6.42 | `4a702be4` |
| turkcorpus | L=0.25 | 21.21 | 3.43 | 1.19 | `f0afe9fd` |
| turkcorpus | L=0.50 | 35.06 | 50.70 | 4.97 | `5e26cf17` |
| turkcorpus | L=0.75 | 39.23 | 76.42 | 6.67 | `24d2fa8a` |
| turkcorpus | L=0.80 | 40.60 | 85.59 | 7.53 | `5e1090c1` |
| turkcorpus | L=0.85 | 40.91 | 85.45 | 7.55 | `aecec72c` |
| turkcorpus | L=0.90 | 36.07 | 93.93 | 9.15 | `df6d6b29` |
| turkcorpus | L=0.95 | 36.41 | 93.25 | 9.08 | `6805dee7` |
| turkcorpus | L=1.00 | 35.24 | 57.35 | 5.51 | `91fde837` |
| turkcorpus | L=1.05 | 39.08 | 75.72 | 6.58 | `9787cb78` |
| turkcorpus | L=1.10 | 36.58 | 63.59 | 5.78 | `655a7da7` |
| turkcorpus | L=1.15 | 38.43 | 72.58 | 6.31 | `4a702be4` |

### `exp_1611808804849777_WR`

| Test set | Tokens | SARI | BLEU | FKGL | Hash |
| --- | --- | --- | --- | --- | --- |
| asset | WR=0.25 | 41.41 | 76.75 | 5.86 | `617eefc8` |
| asset | WR=0.50 | 41.11 | 78.42 | 5.97 | `23a533aa` |
| asset | WR=0.75 | 40.76 | 85.97 | 7.33 | `5b2d7b84` |
| asset | WR=0.80 | 37.52 | 88.64 | 8.43 | `9371b403` |
| asset | WR=0.85 | 38.49 | 88.28 | 8.04 | `fb04e800` |
| asset | WR=0.90 | 34.29 | 89.95 | 8.89 | `30f1709c` |
| asset | WR=0.95 | 35.11 | 89.77 | 8.82 | `7c3bab32` |
| asset | WR=1.00 | 32.85 | 89.56 | 8.92 | `80b5629b` |
| asset | WR=1.05 | 33.69 | 89.39 | 8.82 | `0bac2906` |
| asset | WR=1.10 | 35.28 | 88.63 | 8.40 | `48e43f5e` |
| asset | WR=1.15 | 36.14 | 88.11 | 8.27 | `487a32f0` |
| turkcorpus | WR=0.25 | 33.31 | 41.17 | 4.05 | `617eefc8` |
| turkcorpus | WR=0.50 | 33.40 | 41.67 | 4.01 | `23a533aa` |
| turkcorpus | WR=0.75 | 39.77 | 71.16 | 6.17 | `5b2d7b84` |
| turkcorpus | WR=0.80 | 40.01 | 86.02 | 7.86 | `9371b403` |
| turkcorpus | WR=0.85 | 40.45 | 85.46 | 7.67 | `fb04e800` |
| turkcorpus | WR=0.90 | 37.64 | 91.94 | 8.85 | `30f1709c` |
| turkcorpus | WR=0.95 | 38.35 | 91.56 | 8.73 | `7c3bab32` |
| turkcorpus | WR=1.00 | 35.74 | 92.44 | 9.10 | `80b5629b` |
| turkcorpus | WR=1.05 | 35.98 | 91.26 | 8.97 | `0bac2906` |
| turkcorpus | WR=1.10 | 37.01 | 88.89 | 8.54 | `48e43f5e` |
| turkcorpus | WR=1.15 | 37.17 | 87.62 | 8.33 | `487a32f0` |

## Footnote

Rows in the file: 241. Rows matched to a token setting: 241. Rows dropped: 0.
Command: `python scripts/make_docs_tables.py ablations --d8 /Volumes/D8/workspace`.
