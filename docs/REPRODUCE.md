# Reproducing

## Control-token features vs 2021 processed data

`tests/test_features.py::test_encode_pair_matches_2021_processed_data` is marked xfail.
On the 30-row fixture, `C_`, `L_` and `WR_` match exactly. `DTD_` (dependency tree depth)
differs on 11/30 rows. Cause: the 2021 data was produced with spaCy 2.x `en_core_web_sm`;
spaCy 3.x models parse differently (model 3.0.0: 10/30 mismatches, 3.1.0: 8/30, 3.7.1: 11/30,
3.8.0: 11/30). spaCy 2 models cannot be loaded by spaCy 3. Inference does not depend on this:
it uses `encode_sentence` with fixed target ratios.

## Scoring (easse)

easse is pinned to commit `6a4352ec299ed03fda8ee45445ca43d9c7673e89` (git HEAD at release time) in `pyproject.toml`.
`tests/test_scoring.py` re-scores the published `system_outputs/` files before any model runs,
which separates easse drift from model drift. HEAD reproduces the README within the gate
(SARI +-0.1, BLEU/FKGL +-0.5), so the 2021 commit `8df369f025a7c927425482e802fb546fd5d06d84`
was not needed. If a later easse moves these numbers, pin that commit instead.

| Model | Test set | SARI | BLEU | FKGL |
|---|---|---|---|---|
| 4-token | TurkCorpus | 43.50 | 67.26 | 6.17 |
| 4-token | ASSET | 45.05 | 72.24 | 6.33 |
| 5-token | TurkCorpus | 43.34 | 65.54 | 5.55 |
| 5-token | ASSET | 45.26 | 72.19 | 5.92 |

Scores above are from easse 6a4352ec299ed03fda8ee45445ca43d9c7673e89; each matches the README to 2 decimals.

## Reproduction gate (local, CPU, no uploads)

**Released weights: the Drive (2020) runs at `max_seq_length` 256**
(`experiments/hf/{4,5}tokens`, `config.ts_t5.max_seq_length` 256). They reproduce the README and `system_outputs/`.
SHA-256 of their `pytorch_model.bin`: 4-token `e693d5bdaefdd614b441bb5c25b1d88725d71d698d5727c960c0d8ea7e8b5f65`,
5-token `36091a14fbb5dfd0b0ecea20304df319f7ff92ae1044f120322461bf929d844f`. The D8 tables below describe a later
2021 run that does not reproduce the README (kept as `experiments/hf_d8`). `convert_bin` drops the unused stray key
and the converted dirs load with no missing/unexpected keys.

Result: **the D8 (2021) weights do not reproduce the README; the Google Drive weights do, but only at
`max_seq_length` 256, not 80.** Gate margins: SARI +-0.1, BLEU/FKGL +-0.5. All runs: `ts-t5 evaluate --device cpu`,
explicit tokens (4-token `C=0.95,L=0.75,WR=0.75,DTD=0.75`; 5-token adds `W=1.05`), beam 8, easse 6a4352e.

### Weight hashes (SHA-256 of `pytorch_model.bin`)

| Source | File | SHA-256 |
|---|---|---|
| Drive 4-token zip (`19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE`) | `model_4tokens/pytorch_model.bin` | `e693d5bdaefdd614b441bb5c25b1d88725d71d698d5727c960c0d8ea7e8b5f65` |
| D8 4-token (`exp_1622148499982082_4tokens`) | `pytorch_model.bin` | `2529e5832b8072db6386d358d8e888e73a2c19536c3bb0c24185ebba0f92c1d1` |
| Drive 5-token zip (`19s68NxHybYta0t-smhWKrGZxUYyRlgHA`) | `model_5tokens/pytorch_model.bin` | `36091a14fbb5dfd0b0ecea20304df319f7ff92ae1044f120322461bf929d844f` |
| D8 5-token (`exp_1623415175800762_5tokens`) | no `pytorch_model.bin`, only Lightning `.ckpt` | n/a |

Drive and D8 4-token files differ. They are different training runs: the Drive files are dated 2020 and their
`params.json` has `max_seq_length` 256, batch 6 x accumulation 16; the D8 run has `max_seq_length` 80, batch 32 x 4,
trained 2021. The D8 copy matches its own copy in `experiments/d8/` (same hash).

### Conversion

- D8 4-token: `ts_t5.train.convert_bin` (new helper, wraps a plain `pytorch_model.bin`), tokenizer t5-base,
  `config.ts_t5 = {max_seq_length: 80, features_kwargs: FEATURES_4TOKENS}`.
- D8 5-token: `scripts/convert_ckpt_to_hf.py` path (`ts_t5.train.convert`) on `checkpoint-epoch=4-val_loss=0.83.ckpt`
  (and `epoch=3` after epoch 4 failed), `features_kwargs: FEATURES_5TOKENS`, `max_seq_length` 80.
- Drive: `convert_bin` after dropping one stray key,
  `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight`, which an older transformers version saved
  and which the current T5 class does not have (it is unused). `max_seq_length` set to 256.

### Scores (SARI / BLEU / FKGL, lowercased; margin check vs README)

D8 weights, `max_seq_length` 80 (the original plan): **FAIL**

| Weights | Test set | Got | Expected | Delta | Gate | Identical lines |
|---|---|---|---|---|---|---|
| D8 4-token | TurkCorpus | 43.40 / 68.44 / 6.17 | 43.50 / 67.26 / 6.17 | -0.10 / +1.18 / 0.00 | FAIL (BLEU) | 65/359 |
| D8 4-token | ASSET | 45.30 / 74.29 / 6.18 | 45.05 / 72.24 / 6.33 | +0.25 / +2.05 / -0.15 | FAIL | 62/359 |
| D8 5-token epoch=4 | TurkCorpus | 43.59 / 66.56 / 5.60 | 43.34 / 65.54 / 5.55 | +0.25 / +1.02 / +0.05 | FAIL | 64/359 |
| D8 5-token epoch=4 | ASSET | 45.32 / 72.85 / 5.82 | 45.26 / 72.19 / 5.92 | +0.06 / +0.66 / -0.10 | FAIL (BLEU) | 78/359 |
| D8 5-token epoch=3 | TurkCorpus | 43.57 / 68.09 / 5.78 | 43.34 / 65.54 / 5.55 | +0.23 / +2.55 / +0.23 | FAIL | 63/359 |
| D8 5-token epoch=3 | ASSET | 44.78 / 74.54 / 5.87 | 45.26 / 72.19 / 5.92 | -0.48 / +2.35 / -0.05 | FAIL | 69/359 |

Drive weights, `max_seq_length` 80 (4-token and 5-token, TurkCorpus only): 4-token 43.32 / 67.11 / 6.20
(343/359 identical), 5-token 43.17 / 66.02 / 5.59 (336/359). Close, but SARI misses by 0.18 and 0.17: the differing
lines are the long sources that 80 tokens truncate.

Drive weights, `max_seq_length` 256: **PASS**

| Weights | Test set | Got | Expected | Delta | Gate | Identical lines |
|---|---|---|---|---|---|---|
| Drive 4-token | TurkCorpus | 43.49 / 67.23 / 6.17 | 43.50 / 67.26 / 6.17 | -0.01 / -0.03 / 0.00 | pass | 358/359 |
| Drive 4-token | ASSET | 45.08 / 72.37 / 6.31 | 45.05 / 72.24 / 6.33 | +0.03 / +0.13 / -0.02 | pass | 355/359 |
| Drive 5-token | TurkCorpus | 43.33 / 65.53 / 5.55 | 43.34 / 65.54 / 5.55 | -0.01 / -0.01 / 0.00 | pass | 357/359 |
| Drive 5-token | ASSET | 45.27 / 72.20 / 5.91 | 45.26 / 72.19 / 5.92 | +0.01 / +0.01 / -0.01 | pass | 358/359 |

Identical-line share is against `system_outputs/TS_T5_*tokens/<hash>_*.test.txt`; output names matched the hashes
`d859b9c4aab9f7a9c719167728399278` and `267a9c5912b4c0e37bf8bf7260f7576d`.

### Which weights, and why

The published outputs come from the Drive weights run with `max_seq_length` 256 (the value in their `params.json`,
and the paper says "maximum token of 256"). The D8 runs are different, later trainings: 18-22% identical lines and
BLEU about 1-2 points high. Release candidates are therefore the Drive weights with `config.ts_t5.max_seq_length = 256`.
The earlier constraint "`max_length` = 80 for both paper models" was wrong for these weights; the release uses 256.
`test_simplify_real_model` now checks `max_len` against the model's `config.ts_t5`.
The D8 weights are a legitimate model, but not the one the README, `system_outputs/` or the paper describe.

### Paper vs README (Sheang and Saggion, INLG 2021, Tables 2 and 5)

| Model | Test set | Paper | README | Difference |
|---|---|---|---|---|
| 4-token | TurkCorpus | 43.31 / 66.23 / 6.17 | 43.50 / 67.26 / 6.17 | +0.19 / +1.03 / 0 |
| 4-token | ASSET | 44.91 / 71.96 / 6.32 | 45.05 / 72.24 / 6.33 | +0.14 / +0.28 / +0.01 |
| 5-token | TurkCorpus | 43.00 / 64.42 / 5.63 | 43.34 / 65.54 / 5.55 | +0.34 / +1.12 / -0.08 |
| 5-token | ASSET | 45.04 / 71.21 / 5.88 | 45.26 / 72.19 / 5.92 | +0.22 / +0.98 / +0.04 |

Every README SARI/BLEU is higher than the paper. Paper Table 6 uses different token values for the 5-token model on
TurkCorpus (`WR0.85 L0.85 DTD0.85`, not 0.75) and the paper tokens were searched per test set; the README and
`system_outputs/` use one 0.75 setting for both sets. README numbers were not edited.

### Slow tests

`TS_T5_MODEL=experiments/hf/4tokens pytest -m slow`: 2 passed (the Drive 4-token weights at 256). The score
reproduction is the table above.
