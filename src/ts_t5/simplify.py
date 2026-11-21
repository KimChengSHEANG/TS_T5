"""Sentence simplification with a TS-T5 model (ported from source/evaluate.py)."""
import json
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from ts_t5.data import write_lines, yield_lines
from ts_t5.features import FEATURES_4TOKENS, Preprocessor

DEFAULT_MAX_LEN = 80
TOKENIZER_FILES = ('tokenizer.json', 'spiece.model', 'tokenizer_config.json')


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Simplifier:
    def __init__(self, model: str | Path, features_kwargs: dict | None = None, device: str | None = None):
        torch.manual_seed(12)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12)
        local = Path(model).is_dir()
        params = {}
        if local and (Path(model) / 'params.json').exists():
            params = json.loads((Path(model) / 'params.json').read_text())
        self.device = _pick_device(device)
        self.model = T5ForConditionalGeneration.from_pretrained(str(model)).to(self.device)
        self.model.eval()
        self.tokenizer = self._load_tokenizer(model, local, params)
        ts_t5 = getattr(self.model.config, 'ts_t5', None) or {}
        self.max_len = int(ts_t5.get('max_seq_length') or params.get('max_seq_length') or DEFAULT_MAX_LEN)
        self.features_kwargs = features_kwargs or ts_t5.get('features_kwargs') or FEATURES_4TOKENS
        self.preprocessor = Preprocessor(self.features_kwargs)

    @staticmethod
    def _load_tokenizer(model, local, params):
        # Legacy local dirs hold only weights: the tokenizer came from params['tokenizer_name_or_path'].
        if local and not any((Path(model) / f).exists() for f in TOKENIZER_FILES):
            return T5TokenizerFast.from_pretrained(params.get('tokenizer_name_or_path', 't5-base'))
        return T5TokenizerFast.from_pretrained(str(model))

    def simplify(self, sentence: str) -> str:
        """Simplified sentence, or '' when the model output equals its (encoded) input."""
        encoded = self.preprocessor.encode_sentence(sentence)
        encoding = self.tokenizer('simplify: ' + encoded, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_tensors='pt')
        beam_outputs = self.model.generate(
            input_ids=encoding['input_ids'].to(self.device),
            attention_mask=encoding['attention_mask'].to(self.device),
            do_sample=False,
            max_length=self.max_len,
            num_beams=8,
            early_stopping=True,
            num_return_sequences=1,
        )
        outputs = []
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != encoded.lower() and sent not in outputs:
                outputs.append(sent)
        return outputs[0] if outputs else ''

    # ponytail: one sentence per generate() call like 2021; batch only after the gate shows identical outputs
    def simplify_file(self, src: Path, dst: Path) -> Path:
        lines = [self.simplify(s) for s in yield_lines(src)]
        write_lines([line.replace("''", '"') for line in lines], dst)
        return Path(dst)
