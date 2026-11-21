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
