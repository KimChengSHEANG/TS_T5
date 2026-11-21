"""Print the markdown tables used in docs/RESULTS.md, docs/ABLATIONS.md, docs/EXPERIMENTS.md and docs/FINDINGS.md.

    python scripts/make_docs_tables.py scores
    python scripts/make_docs_tables.py ablations   --d8 /Volumes/D8/workspace
    python scripts/make_docs_tables.py experiments --d8 /Volumes/D8/workspace
    python scripts/make_docs_tables.py search      --d8 /Volumes/D8/workspace

`scores` needs only the repo. The other commands read the 2021 D8 backup (read-only) and are maintainer tools.
"""
import argparse
import ast
import itertools
import json
import os
import re
import sqlite3
import tempfile
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ts_t5.data import REPO_DIR, generate_hash, read_lines
from ts_t5.features import TOKEN_CLASSES

D8_ROOTS = ['TS_T5/outputs', 'TS_T5/results', 'TS_T5/experiments', 'TS_backup/TS_T5.2/experiments']
ABLATION_FILE = 'TS_T5/results/for_ablations.txt'
EXP_ROOTS = ['TS_T5/experiments', 'TS_backup/TS_T5.2/experiments']
TOKEN_RE = re.compile(r'([A-Z]+):\s*([0-9.e-]+)')
# Values used for the brute-force search of hashes that have no score log (the grid seen in the score logs).
SMALL_GRID = (0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)


def md_row(*cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


# ---------------------------------------------------------------- scores
def cmd_scores(args):
    from ts_t5.evaluate import score_file

    out = REPO_DIR / 'system_outputs'
    jobs = [(p.stem, 'turkcorpus', p) for p in sorted(out.glob('*.txt'))]
    for model in ('TS_T5_4tokens', 'TS_T5_5tokens'):
        for test_set in ('turkcorpus', 'asset'):
            jobs += [(model.replace('TS_T5_', 'TS-T5 '), test_set, p) for p in sorted((out / model).glob(f'*_{test_set}.test.txt'))]
    print(md_row('System', 'Test set', 'SARI', 'BLEU', 'FKGL', 'File'))
    print(md_row('---', '---', '---', '---', '---', '---'))
    for name, test_set, path in jobs:
        n = len(read_lines(path))
        assert n == 359, f'{path}: {n} lines, expected 359'
        s = score_file(test_set, path)
        print(md_row(name, test_set, f"{s['sari']:.2f}", f"{s['bleu']:.2f}", f"{s['fkgl']:.2f}", f'`{path.relative_to(REPO_DIR)}`'))


# ---------------------------------------------------------------- ablations
def collect_score_logs(d8):
    """(path, first line) of every score_*.txt under the D8 roots."""
    rows = []
    for root in D8_ROOTS:
        for d, _, fs in os.walk(Path(d8) / root):
            for f in fs:
                if f.startswith('score_') and f.endswith('.txt'):
                    try:
                        rows.append((f, (Path(d) / f).read_text(errors='replace').split('\n')[0]))
                    except OSError:
                        pass
    return rows


def kwargs_of(pairs):
    return {TOKEN_CLASSES[k]: {'target_ratio': float(v)} for k, v in pairs}


def map_hashes(d8, targets):
    """hash -> [(token, value), ...]. Stage 1: candidates read from the score logs, accepted only if
    generate_hash(candidate) equals the hash. Stage 2: brute force over token subsets, orders and values."""
    found = {}
    grid = set()
    for fname, line in collect_score_logs(d8):
        pairs = TOKEN_RE.findall(line)
        grid.update(float(v) for _, v in pairs)
        if pairs and all(k in TOKEN_CLASSES for k, _ in pairs):
            h = generate_hash(kwargs_of(pairs))
            if h in targets:
                found[h] = [(k, float(v)) for k, v in pairs]
    found.setdefault(generate_hash({}), [])  # the no-token models
    names = list(TOKEN_CLASSES)

    def attempt(keys, vals):
        h = generate_hash({TOKEN_CLASSES[k]: {'target_ratio': v} for k, v in zip(keys, vals)})
        if h in targets and h not in found:
            found[h] = list(zip(keys, vals))

    full = sorted(grid)
    for n, values in ((1, full), (2, full), (3, full), (4, SMALL_GRID), (5, SMALL_GRID)):
        if not targets - set(found):
            break
        for keys in itertools.permutations(names, n):
            for vals in itertools.product(values, repeat=n):
                attempt(keys, vals)
    return found


def parse_ablations(path):
    rows, cur = [], None
    for ln in Path(path).read_text(errors='replace').splitlines():
        m = re.match(r'\s*(\w+_test)\s+(\S+)/outputs/([0-9a-f]{32})_(\w+)\.test\.txt', ln)
        if m:
            cur = m.groups()
        elif ln.startswith('{') and cur:
            d = ast.literal_eval(ln)
            rows.append({'set': cur[0].replace('_test', ''), 'exp': cur[1], 'hash': cur[2], **d})
            cur = None
    return rows


def fmt_tokens(pairs):
    return ' '.join(f'{k}={v:.2f}' for k, v in pairs) if pairs else '(no tokens)'


def cmd_ablations(args):
    rows = parse_ablations(Path(args.d8) / ABLATION_FILE)
    targets = {r['hash'] for r in rows}
    found = map_hashes(args.d8, targets)
    matched = [r for r in rows if r['hash'] in found]
    dropped = [r for r in rows if r['hash'] not in found]
    print(f'<!-- rows {len(rows)}, distinct hashes {len(targets)}, matched rows {len(matched)}, '
          f'dropped rows {len(dropped)} ({len({r["hash"] for r in dropped})} hashes) -->\n')
    by_exp = defaultdict(list)
    for r in matched:
        by_exp[r['exp']].append(r)
    for exp in sorted(by_exp):
        print(f'### `exp_{exp}`\n')
        print(md_row('Test set', 'Tokens', 'SARI', 'BLEU', 'FKGL', 'Hash'))
        print(md_row('---', '---', '---', '---', '---', '---'))
        for r in sorted(by_exp[exp], key=lambda r: (r['set'], [(k, v) for k, v in found[r['hash']]])):
            print(md_row(r['set'], fmt_tokens(found[r['hash']]), f"{r['sari']:.2f}", f"{r['bleu']:.2f}", f"{r['fkgl']:.2f}", f"`{r['hash'][:8]}`"))
        print()
    print(f'Rows in the file: {len(rows)}. Matched: {len(matched)}. Dropped (no token setting reproduces the hash): {len(dropped)}.')


# ---------------------------------------------------------------- experiments
ABBR = {'WordRatioFeature': 'W', 'CharRatioFeature': 'C', 'LevenshteinRatioFeature': 'L',
        'WordRankRatioFeature': 'WR', 'DependencyTreeDepthRatioFeature': 'DTD'}


def run_info(d):
    d = Path(d)
    p = json.loads((d / 'params.json').read_text()) if (d / 'params.json').exists() else {}
    t = (d / 'logs.txt').read_text(errors='replace') if (d / 'logs.txt').exists() else None
    fk = p.get('features_kwargs')
    try:
        toks = '+'.join(ABBR[k] for k in ast.literal_eval(fk)) if fk else 'not recorded'
    except (ValueError, SyntaxError, KeyError):
        toks = 'not recorded'
    vl = [float(x) for x in re.findall(r'val_loss=([0-9.]+)', t or '')]
    if not vl and t:
        # older logs have no val_loss in the progress bar: average the per-batch "Val_loss" lines of each validation round
        batches = [float(x) for x in re.findall(r'Val_loss ([0-9.]+)', t)]
        totals = Counter(re.findall(r'Validating:\s+\d+%\|[# ]*\|\s+\d+/(\d+)', t))
        if batches and totals:
            n = int(totals.most_common(1)[0][0])
            vl = [sum(batches[i:i + n]) / n for i in range(0, len(batches) - n + 1, n)]
    run = {
        'id': d.name, 'date': datetime.fromtimestamp(int(re.search(r'\d{16}', d.name).group()) / 1e6, timezone.utc).strftime('%Y-%m-%d'),
        'model': p.get('model_name') or p.get('model_name_or_path') or '-', 'dataset': p.get('dataset', '-'), 'tokens': toks,
        'epochs': p.get('num_train_epochs', '-'), 'seq': p.get('max_seq_length', '-'),
        'batch': f"{p.get('train_batch_size', '-')}x{p.get('gradient_accumulation_steps', '1')}",
        'val_loss': min(vl) if vl else None, 'log': t is not None,
    }
    if t is None:
        run['status'] = 'no logs.txt (params only)'
    elif 'training finished' in t:
        run['status'] = 'finished'
    else:
        run['status'] = 'no "training finished" line'
    return run


def cmd_experiments(args):
    runs, seen = [], {}
    for root in EXP_ROOTS:
        for d, ds, fs in os.walk(Path(args.d8) / root):
            if 'params.json' in fs or 'logs.txt' in fs:
                ds[:] = []
                if Path(d).name in seen:  # same run kept in both backups
                    seen[Path(d).name]['copies'] += 1
                    continue
                runs.append(run_info(d))
                seen[Path(d).name] = runs[-1]
                runs[-1]['copies'] = 1
    sweep = [r for r in runs if r['model'] == 't5-small' and r['seq'] == '80' and int(r['epochs']) <= 3]
    main = sorted((r for r in runs if r not in sweep), key=lambda r: r['id'])
    print(md_row('Run id', 'Date', 'Model', 'Dataset', 'Tokens', 'Epochs', 'max_seq', 'Batch x accum', 'Best val_loss (bar)', 'Status'))
    print(md_row(*['---'] * 10))
    for r in main:
        vl = f"{r['val_loss']:.3f}" if r['val_loss'] is not None else '-'
        print(md_row(f"`{r['id']}`", r['date'], r['model'], r['dataset'], r['tokens'], r['epochs'], r['seq'], r['batch'], vl, r['status']))
    if sweep:
        v = [r['val_loss'] for r in sweep if r['val_loss'] is not None]
        fin = sum(r['status'] == 'finished' for r in sweep)
        print(md_row(f'{len(sweep)} runs `exp_16226...`-`exp_16232...`', f"{min(r['date'] for r in sweep)} to {max(r['date'] for r in sweep)}",
                     't5-small', '-', '+'.join(sorted({r['tokens'] for r in sweep})), '1-3', '80', '64x1', f'{min(v):.3f} to {max(v):.3f}' if v else '-', f'{fin} finished, {len(sweep) - fin} not', ))
    print(f'\nDistinct runs read: {len(runs)} ({sum(r["copies"] > 1 for r in runs)} of them present in both backups)')


# ---------------------------------------------------------------- search (Optuna)
def cmd_search(args):
    src = Path(args.d8) / 'TS_backup/TS_T5.2/experiments/tuning_logs/old/Tokens_study.db'
    tmp = Path(tempfile.mkdtemp()) / 'Tokens_study.db'
    shutil.copy(src, tmp)  # never open the backup in place
    con = sqlite3.connect(tmp)
    trials = con.execute("select t.number, t.datetime_start, v.value from trials t join trial_values v using(trial_id) where t.state='COMPLETE'").fetchall()
    params = defaultdict(dict)
    for n, name, val in con.execute('select t.number, p.param_name, p.param_value from trials t join trial_params p using(trial_id)'):
        params[n][name] = val
    print(f'Complete trials: {len(trials)}; states: {dict(Counter(s for (s,) in con.execute("select state from trials")))}')
    print(f"Window: {min(t[1] for t in trials)} to {max(t[1] for t in trials)}\n")
    cols = ['CharRatio', 'LevenshteinRatio', 'WordRankRatio', 'DepthTreeRatio']
    print(md_row('Rank', 'Trial', 'C', 'L', 'WR', 'DTD', 'SARI'))
    print(md_row(*['---'] * 7))
    for i, (n, _, v) in enumerate(sorted(trials, key=lambda t: -t[2])[:10], 1):
        print(md_row(i, n, *[f'{params[n][c]:.2f}' for c in cols], f'{v:.2f}'))
    print()
    print(md_row('Token', 'Mean SARI of top 20 trials (value)', 'Mean value, top 20', 'Mean value, all'))
    print(md_row('---', '---', '---', '---'))
    top = sorted(trials, key=lambda t: -t[2])[:20]
    for c in cols:
        print(md_row(c, '', f"{sum(params[n][c] for n, _, _ in top) / 20:.2f}", f"{sum(params[n][c] for n, _, _ in trials) / len(trials):.2f}"))
    near = [(n, v) for n, _, v in trials if all(abs(params[n][c] - d) < 1e-6 for c, d in zip(cols, (0.95, 0.75, 0.75, 0.75)))]
    print(f'\nTrials with exactly C0.95 L0.75 WR0.75 DTD0.75: {near}')
    print(f"All-trial SARI: min {min(t[2] for t in trials):.2f}, max {max(t[2] for t in trials):.2f}, mean {sum(t[2] for t in trials) / len(trials):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('scores').set_defaults(fn=cmd_scores)
    for name, fn in (('ablations', cmd_ablations), ('experiments', cmd_experiments), ('search', cmd_search)):
        p = sub.add_parser(name)
        p.add_argument('--d8', default='/Volumes/D8/workspace', help='D8 workspace directory (read-only)')
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
