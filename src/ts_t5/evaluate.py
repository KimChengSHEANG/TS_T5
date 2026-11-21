"""Scoring (and later, generation) for the published TS-T5 models."""
from pathlib import Path
from typing import Literal

from easse.cli import evaluate_system_output

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
