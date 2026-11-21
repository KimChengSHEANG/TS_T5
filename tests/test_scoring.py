import pytest

from ts_t5.data import REPO_DIR
from ts_t5.evaluate import EXPECTED, score_file


@pytest.mark.parametrize('model,h', [('4tokens', 'd859b9c4aab9f7a9c719167728399278'),
                                     ('5tokens', '267a9c5912b4c0e37bf8bf7260f7576d')])
@pytest.mark.parametrize('test_set', ['turkcorpus', 'asset'])
def test_published_outputs_score_as_readme(model, h, test_set):
    got = score_file(test_set, REPO_DIR / f'system_outputs/TS_T5_{model}/{h}_{test_set}.test.txt')
    want = EXPECTED[model][test_set]
    assert abs(got['sari'] - want['sari']) <= 0.1
    assert abs(got['bleu'] - want['bleu']) <= 0.5 and abs(got['fkgl'] - want['fkgl']) <= 0.5


@pytest.mark.parametrize('test_set,n', [('turkcorpus', 359), ('asset', 359)])
def test_evaluate_model_reads_existing_source(monkeypatch, tmp_path, test_set, n):
    from ts_t5 import evaluate
    from ts_t5.data import read_lines
    from ts_t5.features import FEATURES_4TOKENS
    from ts_t5.simplify import Simplifier

    seen = {}

    class Stub(Simplifier):
        def __init__(self):
            self.features_kwargs = FEATURES_4TOKENS

        def simplify_file(self, src, out):
            seen['n'] = len(read_lines(src))

    monkeypatch.setattr(evaluate, 'score_file', lambda *a: {})
    evaluate.evaluate_model(Stub(), test_set, None, tmp_path)
    assert seen['n'] == n
