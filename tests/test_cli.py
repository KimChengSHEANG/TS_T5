import pytest

from ts_t5.cli import main, parse_tokens
from ts_t5.features import FEATURES_4TOKENS, FEATURES_5TOKENS


def test_cli_tokens_keep_order():
    assert list(parse_tokens('W=1.05,C=0.95,L=0.75,WR=0.75,DTD=0.75')) == list(FEATURES_5TOKENS)
    assert parse_tokens('C=0.95,L=0.75,WR=0.75,DTD=0.75') == FEATURES_4TOKENS
    assert list(parse_tokens('DTD=0.75,C=0.95')) == ['DependencyTreeDepthRatioFeature', 'CharRatioFeature']


@pytest.mark.parametrize('bad', ['X=1.0', 'C=0.9,C=0.8', 'C', 'C=abc', ''])
def test_cli_tokens_reject_bad(bad):
    with pytest.raises(ValueError):
        parse_tokens(bad)


def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(['--help'])
    out = capsys.readouterr().out
    assert all(c in out for c in ('train', 'evaluate', 'simplify', 'search'))


def test_cli_simplify_dispatch(monkeypatch, capsys):
    calls = {}

    class Fake:
        def __init__(self, model, features_kwargs=None, device=None):
            calls.update(model=model, fk=features_kwargs, device=device)

        def simplify(self, s):
            return s.upper()

    monkeypatch.setattr('ts_t5.cli.Simplifier', Fake)
    assert main(['simplify', '--model', 'm', '--device', 'cpu', 'hello']) == 0
    assert capsys.readouterr().out.strip() == 'HELLO'
    assert calls == {'model': 'm', 'fk': None, 'device': 'cpu'}
