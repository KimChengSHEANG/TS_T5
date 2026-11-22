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


def test_simplify_returns_a_copy_instead_of_an_empty_line():
    import torch

    from ts_t5.features import FEATURES_4TOKENS, Preprocessor

    class Copying:  # no weights: a model whose best beam decodes to the source
        def generate(self, **kwargs):
            return [0]

        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.zeros(1), 'attention_mask': torch.zeros(1)}

        def decode(self, *args, **kwargs):
            return 'A cat .'

    s = Simplifier.__new__(Simplifier)
    s.model = s.tokenizer = Copying()
    s.max_len, s.device, s.preprocessor = 256, 'cpu', Preprocessor(FEATURES_4TOKENS)
    assert s.simplify('A cat .') == 'A cat .'


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


def test_config_tokens_reload_in_canonical_order(tmp_path, monkeypatch):
    """config.json is saved with sort_keys, so features_kwargs reload alphabetical; Simplifier must restore W,C,L,WR,DTD."""
    import json

    from transformers import T5Config, T5ForConditionalGeneration

    from ts_t5.features import FEATURES_5TOKENS
    cfg = T5Config(vocab_size=32, d_model=8, d_kv=4, d_ff=8, num_layers=1, num_heads=2)
    cfg.ts_t5 = {'max_seq_length': 256, 'features_kwargs': dict(FEATURES_5TOKENS)}
    T5ForConditionalGeneration(cfg).save_pretrained(tmp_path)
    saved = json.loads((tmp_path / 'config.json').read_text())['ts_t5']['features_kwargs']
    assert list(saved) == sorted(saved) != list(FEATURES_5TOKENS)  # the hazard is real
    monkeypatch.setattr(Simplifier, '_load_tokenizer', staticmethod(lambda *a: None))
    s = Simplifier(tmp_path, device='cpu')
    assert s.preprocessor.encode_sentence('x') == 'W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  x'


def test_explicit_tokens_keep_given_order(tmp_path, monkeypatch):
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=32, d_model=8, d_kv=4, d_ff=8, num_layers=1, num_heads=2)
    T5ForConditionalGeneration(cfg).save_pretrained(tmp_path)
    monkeypatch.setattr(Simplifier, '_load_tokenizer', staticmethod(lambda *a: None))
    fk = {'LevenshteinRatioFeature': {'target_ratio': 0.5}, 'CharRatioFeature': {'target_ratio': 0.9}}
    assert Simplifier(tmp_path, fk, device='cpu').preprocessor.encode_sentence('x') == 'L_0.50 C_0.90  x'
