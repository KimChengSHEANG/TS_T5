"""Optuna search for control-token values (ported from source/find_best_tokens.py)."""
from pathlib import Path

from ts_t5.data import ASSET, TURKCORPUS, get_data_filepath, read_lines
from ts_t5.features import Preprocessor
from ts_t5.simplify import Simplifier

TOKEN_CLASSES = {
    'W': 'WordRatioFeature',
    'C': 'CharRatioFeature',
    'L': 'LevenshteinRatioFeature',
    'WR': 'WordRankRatioFeature',
    'DTD': 'DependencyTreeDepthRatioFeature',
}
# Legacy search space: trial.suggest_float(name, 0.20, 1.5, step=0.05)
LOW, HIGH, STEP = 0.20, 1.5, 0.05


def _refs(test_set: str, phase: str) -> tuple[str, list]:
    if test_set == TURKCORPUS:
        return 'complex', [get_data_filepath(test_set, phase, 'simple.turk', i) for i in range(8)]
    if test_set == ASSET:
        return 'orig', [get_data_filepath(test_set, phase, 'simp', i) for i in range(10)]
    raise ValueError(f'unknown test_set {test_set!r}')


def search(model: str | Path, n_trials: int, test_set: str = 'turkcorpus', phase: str = 'valid',
           tokens: tuple[str, ...] = ('C', 'L', 'WR', 'DTD'), limit: int | None = None,
           device: str | None = None) -> dict:
    """Maximise corpus SARI over token values; returns {'features_kwargs': ..., 'sari': ...}."""
    import optuna
    from easse.sari import corpus_sari

    unknown = [t for t in tokens if t not in TOKEN_CLASSES]
    if unknown:
        raise ValueError(f'unknown tokens {unknown}; choose from {list(TOKEN_CLASSES)}')
    tokens = tuple(t for t in TOKEN_CLASSES if t in tokens)  # canonical W,C,L,WR,DTD order; dedupes
    if not tokens:
        raise ValueError('tokens must not be empty')
    src_type, ref_paths = _refs(test_set, phase)
    orig = read_lines(get_data_filepath(test_set, phase, src_type))[:limit]
    refs = [read_lines(p)[:limit] for p in ref_paths]
    assert all(len(r) == len(orig) for r in refs), 'reference lengths differ from source length'

    simplifier = Simplifier(model, {TOKEN_CLASSES[t]: {'target_ratio': 1.0} for t in tokens}, device=device)

    def objective(trial) -> float:
        features_kwargs = {TOKEN_CLASSES[t]: {'target_ratio': round(trial.suggest_float(t, LOW, HIGH, step=STEP), 2)}
                           for t in tokens}
        # reuse the loaded weights; only the preprocessor changes per trial
        simplifier.features_kwargs = features_kwargs
        simplifier.preprocessor = Preprocessor(features_kwargs)
        sys_sents = [simplifier.simplify(s).replace("''", '"') for s in orig]
        # Deliberate deviation: legacy generated on `phase` but scored with evaluate_system_output(
        # test_set='turkcorpus_test'), i.e. valid predictions against test refs. Here valid refs are used.
        return corpus_sari(orig, sys_sents, refs, lowercase=True)

    study = optuna.create_study(study_name='Tokens_study', direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best = {TOKEN_CLASSES[t]: {'target_ratio': round(study.best_trial.params[t], 2)} for t in tokens}
    return {'features_kwargs': best, 'sari': study.best_value}
