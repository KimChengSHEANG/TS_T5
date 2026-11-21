"""Scoring (and later, generation) for the published TS-T5 models."""
from pathlib import Path
from typing import Literal

from easse.cli import evaluate_system_output

from ts_t5.data import generate_hash, get_data_filepath
from ts_t5.features import Preprocessor
from ts_t5.simplify import Simplifier

# README table, 4-token and 5-token paper models.
EXPECTED = {
    '4tokens': {
        'turkcorpus': {'sari': 43.50, 'bleu': 67.26, 'fkgl': 6.17},
        'asset': {'sari': 45.05, 'bleu': 72.24, 'fkgl': 6.33},
    },
    '5tokens': {
        'turkcorpus': {'sari': 43.34, 'bleu': 65.54, 'fkgl': 5.55},
        'asset': {'sari': 45.26, 'bleu': 72.19, 'fkgl': 5.92},
    },
}


def score_file(test_set: Literal['turkcorpus', 'asset'], sys_path: Path) -> dict[str, float]:
    """SARI/BLEU/FKGL of a system output file, lowercased, via easse."""
    scores = evaluate_system_output(test_set=f'{test_set}_test', sys_sents_path=str(sys_path), lowercase=True)
    return {k: scores[k] for k in ('sari', 'bleu', 'fkgl')}


def evaluate_model(model, test_set: Literal['turkcorpus', 'asset'], features_kwargs: dict | None, out_dir: Path,
                   device: str | None = None) -> dict[str, float]:
    """Simplify the <test_set> test sources and score the output.

    `model` is a model id/path or an existing Simplifier (reused as is; features_kwargs=None keeps its own).
    """
    if isinstance(model, Simplifier):
        simplifier = model
        if features_kwargs is not None:
            simplifier.features_kwargs = features_kwargs
            simplifier.preprocessor = Preprocessor(features_kwargs)
    else:
        simplifier = Simplifier(model, features_kwargs, device=device)
    features_kwargs = simplifier.features_kwargs
    src = get_data_filepath(test_set, 'test', 'complex')
    out = Path(out_dir) / f'{generate_hash(features_kwargs)}_{test_set}.test.txt'
    simplifier.simplify_file(src, out)
    return score_file(test_set, out)
