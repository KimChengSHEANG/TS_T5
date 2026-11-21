"""Training entry point and Lightning checkpoint -> HF conversion."""
import ast
import json
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoConfig, T5ForConditionalGeneration, T5TokenizerFast

from ts_t5.data import EXP_DIR, REPO_DIR, get_data_filepath, read_lines, write_lines
from ts_t5.features import Preprocessor
from ts_t5.model import T5FineTuner

CONFIG_DIR = REPO_DIR / 'configs'
# tied to lm_head in T5; absent from a stripped state dict is fine
TOLERATED_MISSING = {'lm_head.weight'}


def load_config(name_or_path) -> dict:
    """Read configs/<name>.json, or any JSON file path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f'{name_or_path}.json'
    return json.loads(p.read_text())


def _train_files(config: dict, preprocessor: Preprocessor, output_dir: Path):
    dataset = config['dataset']
    limit = config.get('max_train_pairs')
    if not limit:
        out_dir = preprocessor.preprocess_dataset(dataset)
        return (out_dir / get_data_filepath(dataset, 'train', 'complex').name,
                out_dir / get_data_filepath(dataset, 'train', 'simple').name)
    # smoke runs: encode only the first `limit` pairs instead of the whole train set
    sub = output_dir / 'subset'
    raw_c, raw_s = sub / 'train.complex', sub / 'train.simple'
    write_lines(read_lines(get_data_filepath(dataset, 'train', 'complex'))[:limit], raw_c)
    write_lines(read_lines(get_data_filepath(dataset, 'train', 'simple'))[:limit], raw_s)
    enc_c, enc_s = sub / 'train.complex.encoded', sub / 'train.simple.encoded'
    write_lines(preprocessor.encode_file_pair(raw_c, raw_s), enc_c)
    write_lines(read_lines(raw_s), enc_s)
    return enc_c, enc_s


def train(config: dict) -> Path:
    config = dict(config)
    output_dir = Path(config.get('output_dir') or EXP_DIR / f'exp_{int(time.time() * 1e6)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = output_dir
    (output_dir / 'params.json').write_text(json.dumps(config, indent=2, default=str))

    L.seed_everything(config['seed'])
    files = _train_files(config, Preprocessor(config['features_kwargs']), output_dir)
    model = T5FineTuner(**{**config, 'train_files': [str(f) for f in files]})

    checkpoint = ModelCheckpoint(dirpath=output_dir, filename='checkpoint-{epoch}', monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=True)
    # fp_16 / opt_level / n_gpu / amp / max_grad_norm in old params.json are dead: not passed on
    trainer = L.Trainer(
        accelerator='auto', devices=1,
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        max_epochs=config['num_train_epochs'],
        max_steps=config.get('max_steps', -1),
        callbacks=[checkpoint],
        logger=CSVLogger(save_dir=output_dir, name='', version=''),
        log_every_n_steps=1,
        num_sanity_val_steps=config.get('nb_sanity_val_steps', 0),
        default_root_dir=output_dir,
    )
    trainer.fit(model)
    return output_dir


def convert(ckpt: Path, out_dir: Path, params: dict) -> Path:
    """Lightning .ckpt -> HF dir: model.safetensors, tokenizer, config.ts_t5."""
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']  # trusted local file
    return save_hf(({k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}), out_dir, params)


def convert_bin(bin_path: Path, out_dir: Path, params: dict) -> Path:
    """Plain T5ForConditionalGeneration state dict (pytorch_model.bin) -> HF dir, same layout as `convert`."""
    state = torch.load(bin_path, map_location='cpu', weights_only=True)
    # transformers 3.x also saved the decoder block-0 cross-attention relative_attention_bias, which T5 never
    # uses (only self-attention biases are), so dropping it changes no output
    state.pop('decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight', None)
    return save_hf(state, out_dir, params)


def save_hf(state: dict, out_dir: Path, params: dict) -> Path:
    """Load a T5 state dict into params['model_name'] and save it with tokenizer and config.ts_t5."""
    out_dir = Path(out_dir)
    model = T5ForConditionalGeneration(AutoConfig.from_pretrained(params['model_name']))
    result = model.load_state_dict(state, strict=False)
    missing = sorted(set(result.missing_keys) - TOLERATED_MISSING)
    unexpected = sorted(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(f'checkpoint does not match {params["model_name"]}: '
                         f'missing {missing[:5]} ({len(missing)} total), unexpected {unexpected[:5]} ({len(unexpected)} total)')
    model.tie_weights()
    features_kwargs = params['features_kwargs']
    if isinstance(features_kwargs, str):  # legacy params.json stored a Python repr
        features_kwargs = ast.literal_eval(features_kwargs)
    model.config.ts_t5 = {'max_seq_length': int(params['max_seq_length']), 'features_kwargs': features_kwargs}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    T5TokenizerFast.from_pretrained(params['model_name']).save_pretrained(out_dir)
    return out_dir
