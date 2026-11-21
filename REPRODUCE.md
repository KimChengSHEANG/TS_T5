# Reproducing

## Control-token features vs 2021 processed data

`tests/test_features.py::test_encode_pair_matches_2021_processed_data` is marked xfail.
On the 30-row fixture, `C_`, `L_` and `WR_` match exactly. `DTD_` (dependency tree depth)
differs on 11/30 rows. Cause: the 2021 data was produced with spaCy 2.x `en_core_web_sm`;
spaCy 3.x models parse differently (model 3.0.0: 10/30 mismatches, 3.1.0: 8/30, 3.7.1: 11/30,
3.8.0: 11/30). spaCy 2 models cannot be loaded by spaCy 3. Inference does not depend on this:
it uses `encode_sentence` with fixed target ratios.
