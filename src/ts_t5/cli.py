"""`ts-t5` command line: train, evaluate, simplify, search."""
import argparse
import sys
from pathlib import Path

from ts_t5.features import TOKEN_CLASSES
from ts_t5.simplify import Simplifier


def parse_tokens(s: str) -> dict:
    """'C=0.95,L=0.75' -> {'CharRatioFeature': {'target_ratio': 0.95}, ...}, keeping the given order."""
    out = {}
    for item in s.split(','):
        name, sep, value = item.strip().partition('=')
        if not sep:
            raise ValueError(f'bad token {item!r}; expected NAME=VALUE (names: {", ".join(TOKEN_CLASSES)})')
        if name not in TOKEN_CLASSES:
            raise ValueError(f'unknown token {name!r}; choose from {", ".join(TOKEN_CLASSES)}')
        cls = TOKEN_CLASSES[name]
        if cls in out:
            raise ValueError(f'duplicate token {name!r}')
        try:
            out[cls] = {'target_ratio': float(value)}
        except ValueError:
            raise ValueError(f'bad value {value!r} for token {name!r}') from None
    return out


def _features(args) -> dict | None:
    return parse_tokens(args.tokens) if args.tokens else None


def _cmd_train(args) -> int:
    from ts_t5.train import load_config, train
    print(train(load_config(args.config)))
    return 0


def _cmd_evaluate(args) -> int:
    from ts_t5.evaluate import evaluate_model
    from ts_t5.features import FEATURES_4TOKENS
    sets = ('turkcorpus', 'asset') if args.test_set == 'all' else (args.test_set,)
    features = _features(args)
    if features is None:  # score with the model's own defaults, as simplify does
        features = Simplifier(args.model, None, device=args.device).features_kwargs or FEATURES_4TOKENS
    rows = {t: evaluate_model(args.model, t, features, Path(args.out)) for t in sets}
    print(f'{"test set":<12}{"SARI":>8}{"BLEU":>8}{"FKGL":>8}')
    for t, r in rows.items():
        print(f'{t:<12}{r["sari"]:>8.2f}{r["bleu"]:>8.2f}{r["fkgl"]:>8.2f}')
    return 0


def _cmd_simplify(args) -> int:
    if bool(args.sentence) == bool(args.input):
        print('error: give either SENTENCE or --input/--output', file=sys.stderr)
        return 2
    if args.input and not args.output:
        print('error: --input requires --output', file=sys.stderr)
        return 2
    simplifier = Simplifier(args.model, _features(args), device=args.device)
    if args.input:
        print(simplifier.simplify_file(args.input, args.output))
    else:
        print(simplifier.simplify(args.sentence))
    return 0


def _cmd_search(args) -> int:
    from ts_t5.search import search
    tokens = tuple(a for a in args.search_tokens.split(',') if a)
    result = search(args.model, args.trials, tokens=tokens, limit=args.limit, device=args.device)
    print(f'best SARI {result["sari"]:.2f}')
    for cls, kw in result['features_kwargs'].items():
        print(f'{cls}: {kw["target_ratio"]}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ts-t5', description='TS-T5 text simplification with control tokens.')
    sub = p.add_subparsers(dest='command', required=True)

    def model_args(sp, tokens=True):
        sp.add_argument('--model', required=True, help='HF repo id or local model directory')
        sp.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default=None, help='default: auto')
        if tokens:
            sp.add_argument('--tokens', help='e.g. C=0.95,L=0.75 (default: the model config)')

    t = sub.add_parser('train', help='fine-tune a model from a config')
    t.add_argument('--config', required=True, help='config name or path')
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser('evaluate', help='print SARI/BLEU/FKGL on the test sets')
    model_args(e)
    e.add_argument('--test-set', choices=('turkcorpus', 'asset', 'all'), default='all')
    e.add_argument('--out', default='system_outputs', help='directory for system outputs')
    e.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser('simplify', help='simplify a sentence or a file')
    model_args(s)
    s.add_argument('sentence', nargs='?')
    s.add_argument('--input')
    s.add_argument('--output')
    s.set_defaults(func=_cmd_simplify)

    q = sub.add_parser('search', help='Optuna search for control-token values (needs the optuna extra)')
    model_args(q, tokens=False)
    q.add_argument('--trials', type=int, required=True)
    q.add_argument('--limit', type=int, help='use only the first K validation sentences')
    q.add_argument('--search-tokens', default='C,L,WR,DTD', help='tokens to search, e.g. W,C,L,WR,DTD')
    q.set_defaults(func=_cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
