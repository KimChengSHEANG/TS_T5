from pathlib import Path

import pytest

from ts_t5.data import read_lines
from ts_t5.features import FEATURES_4TOKENS, FEATURES_5TOKENS, Preprocessor

FIXTURE = Path(__file__).parent / 'fixtures' / 'processed_pairs.tsv'


def test_encode_sentence_exact_string():
    s = 'The cat sat on the mat .'
    assert Preprocessor(FEATURES_4TOKENS).encode_sentence(s) == 'C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s
    assert Preprocessor(FEATURES_5TOKENS).encode_sentence(s) == 'W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s


@pytest.mark.xfail(
    strict=False,
    reason='DTD_ differs on 11/30 fixture rows (C_, L_, WR_ all match): 2021 data was built with '
           'spaCy 2.x en_core_web_sm, whose parses differ from spaCy 3.x models (3.0.0: 10/30, '
           '3.1.0: 8/30, 3.7.1 and 3.8.0: 11/30). Inference does not depend on it. See REPRODUCE.md.')
def test_encode_pair_matches_2021_processed_data():
    """Fixture: first 30 wikilarge train pairs + line from the 2021 processed_data dir
    26ebb6aa762eac859c7b417fbb503eb7 (= hash of the 4-token 0.8 kwargs)."""
    p = Preprocessor({k: {'target_ratio': 0.8} for k in FEATURES_4TOKENS})
    rows = [l.split('\t') for l in read_lines(FIXTURE)]
    mismatches = [c for c, s, want in rows if p.encode_sentence_pair(c, s) != want]
    assert not mismatches, mismatches[:3]
