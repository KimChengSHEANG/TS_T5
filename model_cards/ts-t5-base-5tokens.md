---
language: en
license: cc-by-nc-4.0
base_model: t5-base
library_name: transformers
pipeline_tag: text2text-generation
tags:
- text-simplification
- controllable-generation
- sentence-simplification
- t5
---

# TS-T5 base, 5-token model

Controllable sentence simplification with a unified text-to-text transfer transformer. This is the 5-token model of
[Sheang and Saggion (2021), INLG](https://aclanthology.org/2021.inlg-1.38). A T5-base model is fine-tuned to read control tokens that set how much
the output should be shortened, rewritten and lexically simplified.

Code, evaluation and reproduction notes: [https://github.com/KimChengSHEANG/TS_T5](https://github.com/KimChengSHEANG/TS_T5).

These are the weights from the original 2020 Google Drive training runs (max sequence length 256), converted to
`safetensors`. They are not the later 2021 retrains.

## Intended use and limits

- English only, sentence level (one sentence per input), Wikipedia-style text.
- Non-commercial use only (CC BY-NC 4.0).
- Output can drop or change meaning. Check it before relying on it, especially for names, numbers and negation.
- Returns an empty result in the repo's `Simplifier` when the model just copies its input.

## Training data

WikiLarge (`dataset: wikilarge` in the training config). Hardware and total training time were not recorded for the
released runs; see [docs/REPRODUCE.md](https://github.com/KimChengSHEANG/TS_T5/blob/main/docs/REPRODUCE.md).

## Input format

The model input is `"simplify: "` + control tokens + the sentence. The control tokens are written first, and a double
space separates them from the sentence. The default tokens give, for `The cat sat on the mat .`:

```
"simplify: W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  The cat sat on the mat ."
```

## Control tokens

Each token is a target ratio between the simplified output and the source.

| Token | Controls | Default |
|---|---|---|
| `W` | word count ratio | 1.05 |
| `C` | character length ratio | 0.95 |
| `L` | Levenshtein similarity (how much of the source is kept) | 0.75 |
| `WR` | word rank ratio (lexical complexity, rarer words score higher) | 0.75 |
| `DTD` | dependency tree depth ratio (syntactic complexity) | 0.75 |

Use the defaults first. Other values change the output; the effect of each token is in
[docs/FINDINGS.md](https://github.com/KimChengSHEANG/TS_T5/blob/main/docs/FINDINGS.md). At inference the value is written as given (`C_0.95`, not `C_0.950`).

## How to use

```python
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

repo = "kcsheang/ts-t5-base-5tokens"
tokenizer = T5TokenizerFast.from_pretrained(repo)
model = T5ForConditionalGeneration.from_pretrained(repo).eval()

text = "simplify: W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  The cat sat on the mat ."
enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_beams=8,
        do_sample=False,
        early_stopping=True,
        max_length=256,
        num_return_sequences=1,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Replace the sentence after the control tokens with your own. The repo's CLI builds the input for you:
`ts-t5 simplify` (see the GitHub README).

Decoding used for all reported scores: `num_beams=8`, `do_sample=False`, `early_stopping=True`, `max_length=256`,
input padded to 256 tokens. `generation_config.json` in this repo carries the same decoding settings, so a bare
`model.generate(**inputs)` call decodes the same way (padding to 256 aside).

## Evaluation

Scored with `easse` `evaluate_system_output(test_set=..., lowercase=True)` with `turkcorpus_test` and `asset_test`,
default tokens, decoding as above.

| Test set | SARI | BLEU | FKGL |
|---|---|---|---|
| TurkCorpus test | 43.34 | 65.54 | 5.55 |
| ASSET test | 45.26 | 72.19 | 5.92 |

These are the numbers in the GitHub README. The tables in the paper are lower; the gap and what we could and could not
trace is in [docs/RESULTS.md](https://github.com/KimChengSHEANG/TS_T5/blob/main/docs/RESULTS.md).

## Human evaluation

Crowd raters (Amazon Mechanical Turk, 5-point scale, mean over 500 ratings = 100 sentences x 5 raters) scored this
model against three other systems; this is Table 3 of the paper. The rated sentences were drawn from the TurkCorpus test set.

| System | Fluency | Simplicity | Adequacy |
|---|---|---|---|
| YATS | 4.032 | 3.620 | 3.924 |
| DMASS-DCSS | 3.848 | 3.706 | 3.482 |
| MUSS | 4.410 | 4.022 | 4.130 |
| TS-T5 (this model) | 4.308 | 3.990 | 4.180 |

Protocol, anonymised ratings and caveats: [docs/HUMAN_EVALUATION.md](https://github.com/KimChengSHEANG/TS_T5/blob/main/docs/HUMAN_EVALUATION.md).

## Training hyperparameters

From `configs/5tokens.json`:

| Parameter | Value |
|---|---|
| `model_name` | `t5-base` |
| `max_seq_length` | `256` |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.1` |
| `adam_epsilon` | `1e-08` |
| `warmup_steps` | `5` |
| `train_batch_size` | `6` |
| `gradient_accumulation_steps` | `16` |
| `num_train_epochs` | `5` |
| `max_grad_norm` | `1.0` |
| `seed` | `12` |
| `fp_16` | `False` |

## Citation

```bibtex
@inproceedings{sheang-saggion-2021-controllable,
    title = "Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer",
    author = "Sheang, Kim Cheng  and Saggion, Horacio",
    booktitle = "Proceedings of the 14th International Conference on Natural Language Generation",
    month = aug,
    year = "2021",
    address = "Aberdeen, Scotland, UK",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.inlg-1.38",
    pages = "341--352"
}
```

## License

CC BY-NC 4.0 for the model and the code.
