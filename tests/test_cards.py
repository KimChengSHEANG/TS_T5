import json
import re

import pytest

from ts_t5.data import REPO_DIR
from ts_t5.evaluate import EXPECTED
from ts_t5.features import FEATURES_4TOKENS, FEATURES_5TOKENS, Preprocessor

CARDS = [('4tokens', FEATURES_4TOKENS), ('5tokens', FEATURES_5TOKENS)]


@pytest.mark.parametrize('name,feats', CARDS)
def test_card_input_example_matches_preprocessor(name, feats):
    card = (REPO_DIR / f'model_cards/ts-t5-base-{name}.md').read_text()
    s = 'The cat sat on the mat .'
    assert '"simplify: ' + Preprocessor(feats).encode_sentence(s) + '"' in card
    assert 'license: cc-by-nc-4.0' in card and 'base_model: t5-base' in card
    for v in EXPECTED[name].values():
        assert f"{v['sari']:.2f}" in card


@pytest.mark.parametrize('name,feats', CARDS)
def test_card_has_no_personal_data(name, feats):
    card = (REPO_DIR / f'model_cards/ts-t5-base-{name}.md').read_text()
    assert not re.search(r'A[0-9A-Z]{12,}', card)
    assert not re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', card)
    assert 'Generated with' not in card


@pytest.mark.parametrize('name,feats', CARDS)
def test_released_model_uses_max_seq_length_256(name, feats):
    cfg = REPO_DIR / 'experiments/hf' / name / 'config.json'
    if not cfg.exists():
        pytest.skip('release model dir not present')
    assert json.loads(cfg.read_text())['ts_t5']['max_seq_length'] == 256


def test_push_dry_run_lists_files_and_stays_offline(tmp_path, capsys, monkeypatch):
    import sys
    sys.path.insert(0, str(REPO_DIR / 'scripts'))
    from push_to_hub import push
    for n in ('model.safetensors', 'config.json', 'tokenizer.json', 'optimizer.ckpt'):
        (tmp_path / n).write_text('x')
    card = tmp_path / 'card.md'
    card.write_text('c')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)  # any import in dry run would fail
    push(tmp_path, 'u/r', card)
    out = capsys.readouterr().out
    assert 'private: True' in out and 'README.md' in out and 'model.safetensors' in out
    assert 'optimizer.ckpt' not in out
