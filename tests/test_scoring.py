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
