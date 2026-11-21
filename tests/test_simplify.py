import os

import pytest

from ts_t5.data import read_lines, write_lines
from ts_t5.simplify import Simplifier


def test_simplify_file_keeps_line_count(tmp_path, monkeypatch):
    s = Simplifier.__new__(Simplifier)  # no weights
    monkeypatch.setattr(s, 'simplify', lambda x: '' if 'same' in x else x.upper(), raising=False)
    src = tmp_path / 'in.txt'
    write_lines(['a', 'same', "it ''s"], src)
    out = s.simplify_file(src, tmp_path / 'out.txt')
    assert read_lines(out) == ['A', '', 'IT "S']


@pytest.mark.slow
def test_simplify_real_model():
    s = Simplifier(os.environ.get('TS_T5_MODEL', 'kcsheang/ts-t5-base-4tokens'))
    src = ['The incumbent was defeated in the election by a substantial margin .',
           'Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose .',
           'The committee postponed the decision until additional evidence could be obtained from the relevant authorities .']
    out = [s.simplify(x) for x in src]
    print(list(zip(src, out)))
    assert s.max_len == int(s.model.config.ts_t5['max_seq_length'])  # 256 for the released Drive weights
    assert all(out) and all(o != x for o, x in zip(out, src))
