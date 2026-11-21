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


def _fake_eval(monkeypatch):
    seen = {'inst': [], 'eval': []}

    class Fake:
        def __init__(self, model, features_kwargs=None, device=None):
            seen['inst'].append((model, features_kwargs, device))

    def fake_eval(model, test_set, features_kwargs, out_dir, device=None):
        seen['eval'].append((model, test_set, out_dir))
        return {'sari': 1.0, 'bleu': 2.0, 'fkgl': 3.0}

    monkeypatch.setattr('ts_t5.cli.Simplifier', Fake)
    monkeypatch.setattr('ts_t5.evaluate.evaluate_model', fake_eval)
    return seen


def test_cli_evaluate_one_simplifier_and_device(monkeypatch, capsys):
    seen = _fake_eval(monkeypatch)
    assert main(['evaluate', '--model', 'org/m', '--device', 'cpu', '--test-set', 'all']) == 0
    assert seen['inst'] == [('org/m', None, 'cpu')]
    assert [e[1] for e in seen['eval']] == ['turkcorpus', 'asset']
    assert seen['eval'][0][0] is seen['eval'][1][0]
    assert str(seen['eval'][0][2]) == 'outputs/m'
    assert 'turkcorpus' in capsys.readouterr().out


def test_cli_evaluate_refuses_system_outputs(monkeypatch, capsys):
    from ts_t5.data import REPO_DIR
    seen = _fake_eval(monkeypatch)
    for out in (REPO_DIR / 'system_outputs', REPO_DIR / 'system_outputs' / 'TS_T5_4tokens'):
        assert main(['evaluate', '--model', 'm', '--out', str(out)]) == 2
    assert 'system_outputs' in capsys.readouterr().err
    assert seen['inst'] == []
