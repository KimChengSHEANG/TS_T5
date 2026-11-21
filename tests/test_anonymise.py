import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from anonymise_human_eval import anonymise, check_no_ids  # noqa: E402


def _write(tmp_path, text):
    p = tmp_path / 'f.txt'
    p.write_text(text)
    return p


def test_anonymiser_leaves_no_worker_id(tmp_path):
    df = pd.DataFrame({'WorkerId': ['A1B2C3D4E5F6G7'], 'AssignmentId': ['3ABCDEF'], 'HITId': ['3XYZ'],
                       'Answer.comment': ['my email is x@y.z'], 'Answer.fluency': [4]})
    out = anonymise(df, salt='s')
    assert 'Answer.comment' not in out and out['WorkerId'][0] != 'A1B2C3D4E5F6G7'
    assert out['WorkerId'][0] == anonymise(df, salt='s')['WorkerId'][0]  # stable
    out.to_csv(tmp_path / 'o.csv'); check_no_ids([tmp_path / 'o.csv'])
    with pytest.raises(ValueError): check_no_ids([_write(tmp_path, 'A1B2C3D4E5F6G7')])


def test_drops_other_columns_and_all_nan_answers():
    df = pd.DataFrame({'WorkerId': ['A1B2C3D4E5F6G7'], 'AcceptTime': ['t'], 'Input.orig_sent_1': ['hi'],
                       'Answer.suggestions': [float('nan')], 'Answer.fluency_11': [3]})
    out = anonymise(df, salt='s')
    assert list(out.columns) == ['WorkerId', 'Input.orig_sent_1', 'Answer.fluency_11']


def test_4token_batch_has_no_worker_link():
    # 5-token rows match the public raw exports by their answers, so a shared worker hash re-identifies 4-token rows.
    path = Path(__file__).resolve().parent.parent / 'human_evaluation/mturk_batches/4tokens_batch_4442003.csv'
    assert 'WorkerId' not in pd.read_csv(path).columns
