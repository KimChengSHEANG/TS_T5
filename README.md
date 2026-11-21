# TS-T5: controllable sentence simplification with T5

Code, weights and reproduction notes for
[Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer](https://aclanthology.org/2021.inlg-1.38)
(Sheang and Saggion, INLG 2021). A T5-base model is fine-tuned on WikiLarge to read control tokens (`C`, `L`, `WR`,
`DTD`, and `W` in the 5-token model) that set how much the output is shortened, rewritten and lexically simplified.

## Quickstart

The weights are on the Hugging Face Hub: [`kcsheang/ts-t5-base-4tokens`](https://huggingface.co/kcsheang/ts-t5-base-4tokens)
and [`kcsheang/ts-t5-base-5tokens`](https://huggingface.co/kcsheang/ts-t5-base-5tokens).

```python
from transformers import T5ForConditionalGeneration, T5TokenizerFast
tok = T5TokenizerFast.from_pretrained("kcsheang/ts-t5-base-4tokens")
model = T5ForConditionalGeneration.from_pretrained("kcsheang/ts-t5-base-4tokens")
inputs = tok("simplify: C_0.95 L_0.75 WR_0.75 DTD_0.75  The cat sat on the mat .", return_tensors="pt")
print(tok.decode(model.generate(**inputs)[0], skip_special_tokens=True))
```

The input is `"simplify: "`, then the control tokens, then a double space, then the sentence. Decoding is beam search with
8 beams (`generation_config.json` in the model repo sets it). The reported scores also pad the input to 256 tokens;
the model cards show the exact call.

## Install

Python 3.10 or newer. Install PyTorch for your platform first (see https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .            # package ts_t5 and the ts-t5 command
pip install -e '.[dev]'     # plus the test dependencies
pip install -e '.[search]'  # plus optuna, for ts-t5 search
```

The package pins its dependencies (`pyproject.toml`). Commands that read datasets (`evaluate`, `train`, `search`) must
run from a repo checkout, because they read `resources/`.

## Commands

`<model>` is a Hub id or a local model directory.

```bash
# Simplify one sentence, or a file (one sentence per line)
ts-t5 simplify --model kcsheang/ts-t5-base-4tokens "The cat sat on the mat ."
ts-t5 simplify --model kcsheang/ts-t5-base-4tokens --input complex.txt --output simple.txt
ts-t5 simplify --model kcsheang/ts-t5-base-4tokens --tokens C=0.8,L=0.9 "The cat sat on the mat ."

# Print SARI, BLEU and FKGL on TurkCorpus and ASSET (--test-set turkcorpus|asset|all)
ts-t5 evaluate --model kcsheang/ts-t5-base-4tokens --test-set all

# Fine-tune from a config in configs/ (4tokens, 5tokens, smoke) or a path
ts-t5 train --config 4tokens

# Optuna search for control-token values (needs the search extra)
ts-t5 search --model kcsheang/ts-t5-base-4tokens --trials 50
```

`--device` selects `cpu`, `cuda` or `mps` (default: automatic). `--tokens` overrides the model's default tokens.

Default tokens: 4-token model `C 0.95, L 0.75, WR 0.75, DTD 0.75`; 5-token model `W 1.05` followed by the same four.

## Results

Default tokens, `easse` with `lowercase=True`, beam 8, `max_seq_length` 256.

| Model | Test set | SARI | BLEU | FKGL |
|---|---|---|---|---|
| 4-token | TurkCorpus | 43.50 | 67.26 | 6.17 |
| 4-token | ASSET | 45.05 | 72.24 | 6.33 |
| 5-token | TurkCorpus | 43.34 | 65.54 | 5.55 |
| 5-token | ASSET | 45.26 | 72.19 | 5.92 |

These are the numbers this repository has always reported, and `ts-t5 evaluate` reproduces them from the released
weights. The tables in the paper are lower; the gap and what we could trace is in [docs/RESULTS.md](docs/RESULTS.md).

The released weights are the original 2020 training runs (the ones behind these numbers), converted to `safetensors`.
The 2021 retrains do not reproduce the table.

Documentation in [docs/](docs/):

- [REPRODUCE.md](docs/REPRODUCE.md): how scores and weights were reproduced, environment, tests.
- [RESULTS.md](docs/RESULTS.md): scores of all systems, paper versus README.
- [FINDINGS.md](docs/FINDINGS.md): what we found while preparing the release.
- [ABLATIONS.md](docs/ABLATIONS.md) and [EXPERIMENTS.md](docs/EXPERIMENTS.md): control-token ablations and the experiment log.
- [HUMAN_EVALUATION.md](docs/HUMAN_EVALUATION.md): protocol and ratings. Anonymised ratings are in
  [human_evaluation/](human_evaluation/); the original HIT designs and result sheets are in `Human evaluation/`.

## Tests

```bash
pip install -e '.[dev]'
pytest                                          # fast tests; slow ones are excluded
TS_T5_MODEL=experiments/hf/4tokens pytest -m slow   # needs converted model weights
```

## Downloads

The old Google Drive links for the model weights are retired. Use the Hugging Face repos above.

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

## Licence

[CC BY-NC 4.0](LICENSE) for the code and the models: non-commercial use only.

## Credits

The preprocessing code was adopted from https://github.com/facebookresearch/access.
