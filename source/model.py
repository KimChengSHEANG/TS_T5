from functools import lru_cache
from pathlib import Path


from easse.sari import corpus_sari
from torch.nn import functional as F
from source.helper import log_stdout, tokenize, yield_sentence_pair, yield_lines, load_preprocessor, read_lines, \
    count_line
import argparse
import os
import logging
import random
import nltk
from source.resources import NEWSELA_DATASET, get_data_filepath, WIKILARGE_DATASET, TURKCORPUS_DATASET, \
    WIKILARGE_WIKIAUTO_DATASET

nltk.download('punkt')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup, AutoConfig, AutoModel
)

class T5FineTuner(pl.LightningModule):
    def __init__(self, model_name, learning_rate, adam_epsilon, custom_loss, weight_decay, dataset,
                 train_batch_size, valid_batch_size, train_sample_size, valid_sample_size, max_seq_length,
                 n_gpu, gradient_accumulation_steps, num_train_epochs, warmup_steps, nb_sanity_val_steps,
                 *args, **kwargs):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.hparams.model_name)
        self.model = self.model.to(self.device)
        self.preprocessor = load_preprocessor()


    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, sentence):
        sentence = self.preprocessor.encode_sentence(sentence)
        text = "simplify: " + sentence

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.hparams.max_seq_length,
            padding='max_length',
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=False,
            max_length=self.hparams.max_seq_length,
            num_beams=8,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        )
        pred_sent = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return pred_sent

    def training_step(self, batch, batch_idx):
        labels = batch["target_ids"]
        # Huggingface’s loss functions are defined to exclude the ID -100 during loss calculations. Therefore, we need to convert all padding token IDs in labels to -100.
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        self.opt.zero_grad()
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask'],
        )

        if self.hparams.custom_loss:
            loss = outputs.loss
            complexity_score = random.randint(0, 100) * 0.01
            # complexity_score = self._custom_step(outputs['logits'])
            # loss = loss * complexity_score
            lambda_ = 0.7
            # loss = lambda_ * loss + (1-lambda_)*complexity_score
            # loss = torch.sqrt(loss + lambda_ * complexity_score)
            loss = loss + complexity_score + lambda_ * (complexity_score - loss)
            
            print(complexity_score)
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            # print(loss)
            return loss
        else:
            loss = outputs.loss
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            return loss

        # loss = outputs.loss
        # logs = {"train_loss": loss}
        # self.logger.experiment.add_scalars('loss', logs, global_step=self.global_step)
        # return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss = self.sari_validation_step(batch)
        # loss = self._step(batch)
        print("Val_loss", loss)
        logs = {"val_loss": loss}
        # self.logger.experiment.add_scalars('loss', logs, global_step=self.global_step)
        # return {"val_loss": torch.tensor(loss)}
        self.log('val_loss', loss, batch_size=self.hparams.valid_batch_size)
        return torch.tensor(loss, dtype=float)

    def sari_validation_step(self, batch):
        def generate(sentence):
            sentence = self.preprocessor.encode_sentence(sentence)
            text = "simplify: " + sentence
            # print("Simplifying: ", text)

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.hparams.max_seq_length,
                padding='max_length',
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_masks = encoding["attention_mask"].to(self.device)

            # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                do_sample=False,
                max_length=self.hparams.max_seq_length,
                num_beams=8,
                top_k=120,
                top_p=0.98,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device)
            # final_outputs = []
            # for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # if sent.lower() != sentence.lower() and sent not in final_outputs:
                # final_outputs.append(sent)
            
            return sent
            # return final_outputs[0]

        pred_sents = []
        for source in batch["source"]:
            pred_sent = generate(source)
            pred_sents.append(pred_sent)

        score = corpus_sari(batch["source"], pred_sents, batch["targets"])
        print("Sari score: ", score)

        return 1 - score / 100

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # optimizer = SAM(optimizer_grouped_parameters, base_optimizer, lr=self.hparams.learning_rate, momentum=0.9)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()
        self.lr_scheduler.step()


    def train_dataloader(self):
        train_dataset = TrainDataset(dataset=self.hparams.dataset,
                                     tokenizer=self.tokenizer,
                                     max_len=self.hparams.max_seq_length,
                                     sample_size=self.hparams.train_sample_size)

        dataloader = DataLoader(train_dataset,
                                batch_size=self.hparams.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4)
        t_total = ((len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                   // self.hparams.gradient_accumulation_steps
                   * float(self.hparams.num_train_epochs)
                   )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = ValDataset(dataset=self.hparams.dataset,
                                 tokenizer=self.tokenizer,
                                 max_len=self.hparams.max_seq_length,
                                 sample_size=self.hparams.valid_sample_size)
        return DataLoader(val_dataset,
                          batch_size=self.hparams.valid_batch_size,
                          num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        # print("init TrainDataset ...")
        preprocessor = load_preprocessor()
        self.source_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'complex')
        self.target_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'simple')

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self):
        self.inputs = read_lines(self.source_filepath)
        self.targets = read_lines(self.target_filepath)

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        source = "simplify: " + self.inputs[index]
        target = self.targets[index]

        tokenized_inputs = self.tokenizer(
            [source],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [target],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                'sources': self.inputs[index], 'targets': [self.targets[index]]}


class ValDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        self.source_filepath = get_data_filepath(dataset, 'valid', 'complex')
        if dataset == NEWSELA_DATASET:
            self.target_filepaths = [get_data_filepath(dataset, 'valid', 'simple')]

        else:  # TURKCORPUS_DATASET as default
            self.target_filepaths = [get_data_filepath(TURKCORPUS_DATASET, 'valid', 'simple.turk', i) for i in range(8)]

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._build()

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        return {"source": self.inputs[index], "targets": self.targets[index]}

    def _build(self):
        self.inputs = []
        self.targets = []

        for source in yield_lines(self.source_filepath):
            self.inputs.append(source)

        self.targets = [[] for _ in range(count_line(self.target_filepaths[0]))]
        for filepath in self.target_filepaths:
            for idx, line in enumerate(yield_lines(filepath)):
                self.targets[idx].append(line)


def train(train_args):
    args = argparse.Namespace(**train_args)
    seed_everything(args.seed)

    print(train_args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        amp_backend='apex',
        # gradient_clip_val=args.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), checkpoint_callback],
        # logger=TensorBoardLogger(f'{args.output_dir}/logs'),
        num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
        # plugins='ddp_sharded',
        # progress_bar_refresh_rate=1,

    )

    print("Initialize model")
    model = T5FineTuner(**train_args)

    trainer = pl.Trainer(**train_params)
    print(" Training model")
    trainer.fit(model)

    print("training finished")

    # print("Saving model")
    # model.model.save_pretrained(args.output_dir)

    # print("Saved model")
