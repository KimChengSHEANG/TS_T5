"""LightningModule and datasets for TS-T5 fine-tuning (ported from source/model.py to Lightning 2)."""
import sys
from pathlib import Path

import lightning as L
import torch
from easse.sari import corpus_sari
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup

from ts_t5.data import count_line, get_data_filepath, read_lines, yield_lines, TURKCORPUS
from ts_t5.features import Preprocessor

NO_DECAY = ('bias', 'LayerNorm.weight')
# fork-based workers are unsafe next to MPS/spaCy on macOS
NUM_WORKERS = 0 if sys.platform == 'darwin' else 4


class TrainDataset(Dataset):
    """Pre-encoded (complex, simple) pairs; `inputs` already carry the control tokens."""

    def __init__(self, source_filepath, target_filepath, tokenizer, max_len=256, sample_size=1):
        self.inputs = read_lines(source_filepath)
        self.targets = read_lines(target_filepath)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_size = sample_size

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def _tokenize(self, text):
        return self.tokenizer([text], truncation=True, max_length=self.max_len,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        src = self._tokenize('simplify: ' + self.inputs[index])
        tgt = self._tokenize(self.targets[index])
        return {'source_ids': src['input_ids'].squeeze(), 'source_mask': src['attention_mask'].squeeze(),
                'target_ids': tgt['input_ids'].squeeze(), 'target_mask': tgt['attention_mask'].squeeze()}


class ValDataset(Dataset):
    """TurkCorpus valid: raw source plus its 8 references."""

    def __init__(self, sample_size=1, max_pairs=None):
        self.inputs = read_lines(get_data_filepath(TURKCORPUS, 'valid', 'complex'))
        paths = [get_data_filepath(TURKCORPUS, 'valid', 'simple.turk', i) for i in range(8)]
        self.targets = [[] for _ in range(count_line(paths[0]))]
        for path in paths:
            for idx, line in enumerate(yield_lines(path)):
                self.targets[idx].append(line)
        self.n = min(max_pairs, len(self.inputs)) if max_pairs else int(len(self.inputs) * sample_size)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {'source': self.inputs[index], 'targets': self.targets[index]}


class T5FineTuner(L.LightningModule):
    def __init__(self, model_name, learning_rate, adam_epsilon, weight_decay, dataset, features_kwargs,
                 train_batch_size, valid_batch_size, train_sample_size, valid_sample_size, max_seq_length,
                 train_files=None, max_valid_pairs=None, **kwargs):
        super().__init__()
        # extra keys (fp_16, n_gpu, ...) are kept in hparams for the record, never used
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.train()  # from_pretrained returns eval mode
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.preprocessor = Preprocessor(features_kwargs)

    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask,
                          decoder_attention_mask=decoder_attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        labels = batch['target_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # HF loss ignores -100
        loss = self(input_ids=batch['source_ids'], attention_mask=batch['source_mask'],
                    labels=labels, decoder_attention_mask=batch['target_mask']).loss
        self.log('train_loss', loss, on_step=True, prog_bar=True, batch_size=len(labels))
        return loss

    def _generate(self, sentence):
        text = 'simplify: ' + self.preprocessor.encode_sentence(sentence)
        enc = self.tokenizer(text, truncation=True, max_length=self.hparams.max_seq_length,
                             padding='max_length', return_tensors='pt')
        out = self.model.generate(input_ids=enc['input_ids'].to(self.device),
                                  attention_mask=enc['attention_mask'].to(self.device),
                                  do_sample=False, max_length=self.hparams.max_seq_length,
                                  num_beams=8, early_stopping=True, num_return_sequences=1)
        return self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def validation_step(self, batch, batch_idx):
        preds = [self._generate(s) for s in batch['source']]
        # default collate transposes the 8 references to [8][batch], the layout corpus_sari wants
        score = corpus_sari(batch['source'], preds, batch['targets'])
        loss = 1 - score / 100
        self.log('val_loss', loss, batch_size=len(preds))
        return torch.tensor(loss)

    def configure_optimizers(self):
        named = list(self.model.named_parameters())
        groups = [
            {'params': [p for n, p in named if not any(nd in n for nd in NO_DECAY)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in named if any(nd in n for nd in NO_DECAY)], 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def train_dataloader(self):
        src, tgt = self.hparams.train_files
        ds = TrainDataset(src, tgt, self.tokenizer, self.hparams.max_seq_length, self.hparams.train_sample_size)
        return DataLoader(ds, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                          num_workers=NUM_WORKERS)

    def val_dataloader(self):
        ds = ValDataset(self.hparams.valid_sample_size, self.hparams.max_valid_pairs)
        return DataLoader(ds, batch_size=self.hparams.valid_batch_size, num_workers=NUM_WORKERS)
