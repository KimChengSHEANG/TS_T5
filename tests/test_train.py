import csv
import json
import time
from statistics import mean

import pytest
import torch

from ts_t5.simplify import Simplifier
from ts_t5.train import convert, load_config, train


def test_load_config_by_name_and_path(tmp_path):
    cfg = load_config('4tokens')
    assert cfg['train_batch_size'] == 32 and cfg['gradient_accumulation_steps'] == 4
    assert cfg['dataset'] == 'wikilarge' and cfg['max_seq_length'] == 80
    assert list(load_config('5tokens')['features_kwargs'])[0] == 'WordRatioFeature'
    p = tmp_path / 'c.json'
    p.write_text('{"a": 1}')
    assert load_config(p) == {'a': 1}


def test_convert_rejects_bad_state_dict(tmp_path):
    torch.save({'state_dict': {'wrong.prefix': torch.zeros(1)}}, tmp_path / 'x.ckpt')
    with pytest.raises(ValueError, match='missing'):
        convert(tmp_path / 'x.ckpt', tmp_path / 'hf', {'model_name': 't5-small', 'max_seq_length': 80, 'features_kwargs': {}})


@pytest.mark.slow
def test_smoke_train_converts_and_loads(tmp_path):
    t0 = time.time()
    out = train({**load_config('smoke'), 'output_dir': tmp_path})
    print(f'train wall time {time.time() - t0:.0f}s')
    loss = [float(r['train_loss']) for r in csv.DictReader(open(out / 'metrics.csv')) if r['train_loss']]
    print('first10', mean(loss[:10]), 'last10', mean(loss[-10:]), 'n', len(loss))
    assert mean(loss[-10:]) < mean(loss[:10])
    hf = convert(next(out.glob('*.ckpt')), tmp_path / 'hf', json.load(open(out / 'params.json')))
    assert isinstance(Simplifier(hf).simplify('This is a fairly long and complicated sentence .'), str)
