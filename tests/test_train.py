import csv
import json
import time
from statistics import mean

import pytest
import torch

from ts_t5.simplify import Simplifier
from ts_t5.train import convert, convert_bin, load_config, train


def test_load_config_by_name_and_path(tmp_path):
    cfg = load_config('4tokens')
    assert cfg['train_batch_size'] == 6 and cfg['gradient_accumulation_steps'] == 16
    assert cfg['dataset'] == 'wikilarge' and cfg['max_seq_length'] == 256
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


def test_convert_bin_roundtrip(tmp_path):
    from transformers import AutoConfig, T5ForConditionalGeneration
    model = T5ForConditionalGeneration(AutoConfig.from_pretrained('t5-small'))
    state = model.state_dict()
    state['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'] = torch.zeros(1)  # stray 2020 key
    torch.save(state, tmp_path / 'pytorch_model.bin')
    params = {'model_name': 't5-small', 'max_seq_length': '80', 'features_kwargs': "{'CharRatioFeature': {'target_ratio': 0.95}}"}
    out = convert_bin(tmp_path / 'pytorch_model.bin', tmp_path / 'hf', params)
    assert (out / 'model.safetensors').exists() and (out / 'tokenizer.json').exists()
    _, info = T5ForConditionalGeneration.from_pretrained(out, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    cfg = json.loads((out / 'config.json').read_text())
    assert cfg['ts_t5'] == {'max_seq_length': 80, 'features_kwargs': {'CharRatioFeature': {'target_ratio': 0.95}}}


def test_converted_dir_has_decoding_defaults(tmp_path):
    from transformers import AutoConfig, T5ForConditionalGeneration
    model = T5ForConditionalGeneration(AutoConfig.from_pretrained('t5-small'))
    torch.save(model.state_dict(), tmp_path / 'pytorch_model.bin')
    params = {'model_name': 't5-small', 'max_seq_length': 256, 'features_kwargs': {'CharRatioFeature': {'target_ratio': 0.95}}}
    out = convert_bin(tmp_path / 'pytorch_model.bin', tmp_path / 'hf', params)
    gen = json.loads((out / 'generation_config.json').read_text())
    assert (gen['num_beams'], gen['early_stopping'], gen['max_length'], gen['num_return_sequences']) == (8, True, 256, 1)
    assert gen['do_sample'] is False
