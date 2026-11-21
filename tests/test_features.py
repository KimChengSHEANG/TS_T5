from pathlib import Path

import pytest

from ts_t5.data import DUMPS_DIR
from ts_t5.features import FEATURES_4TOKENS, FEATURES_5TOKENS, Preprocessor

FIXTURE = Path(__file__).parent / 'fixtures' / 'processed_pairs.tsv'

needs_glove = pytest.mark.skipif(not (DUMPS_DIR / 'glove.42B.300d.pk').exists(),
                                 reason='glove.42B.300d.pk missing')


def _rows():
    with FIXTURE.open(encoding='utf-8') as f:
        return [l.rstrip('\n').split('\t') for l in f]


def _split(line):
    tokens, sentence = line.split('  ', 1)
    return dict(t.rsplit('_', 1) for t in tokens.split(' ')), sentence


def test_encode_sentence_exact_string():
    s = 'The cat sat on the mat .'
    assert Preprocessor(FEATURES_4TOKENS).encode_sentence(s) == 'C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s
    assert Preprocessor(FEATURES_5TOKENS).encode_sentence(s) == 'W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s


# Fixture: first 30 wikilarge train pairs + line from the 2021 processed_data dir
# 26ebb6aa762eac859c7b417fbb503eb7 (= hash of the 4-token 0.8 kwargs).
def _encode_all():
    p = Preprocessor({k: {'target_ratio': 0.8} for k in FEATURES_4TOKENS})
    return [(_split(p.encode_sentence_pair(c, s)), _split(want)) for c, s, want in _rows()]


@needs_glove
def test_encode_pair_c_l_wr_and_sentence_match_2021_processed_data():
    for (got, got_s), (want, want_s) in _encode_all():
        assert got_s == want_s
        for k in ('C', 'L', 'WR'):
            assert got[k] == want[k], (want_s[:60], k)


@needs_glove
@pytest.mark.xfail(
    raises=AssertionError, strict=False,
    reason='DTD_ differs on 11/30 rows: 2021 data used spaCy 2.x en_core_web_sm; spaCy 3 models '
           'parse differently (3.0.0: 10/30, 3.1.0: 8/30, 3.7.1/3.8.0: 11/30). See REPRODUCE.md.')
def test_encode_pair_dtd_matches_2021_processed_data():
    mismatches = [w_s for (g, _), (w, w_s) in _encode_all() if g['DTD'] != w['DTD']]
    assert not mismatches, mismatches[:3]
