"""Anonymise the MTurk human-evaluation export and compute aggregate scores.

    TS_T5_SALT="$(cat ~/.config/ts_t5/salt)" python scripts/anonymise_human_eval.py build SRC [--out human_evaluation]
    python scripts/anonymise_human_eval.py scores [--out human_evaluation]

The salt never lives in the repo: it is read from the environment only.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ID_COLUMNS = ['WorkerId', 'AssignmentId', 'HITId']
WORKER_ID = re.compile(r'A[0-9A-Z]{12,}')
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
MODELS = ['YATS', 'DMASS-DCSS', 'MUSS', 'TS_T5']  # index = Input.simp_sent_id_*
ASPECTS = ['fluency', 'simplicity', 'adequacy']


def _hash(value, salt):
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()[:12]


def anonymise(df, salt):
    """Hash the ID columns, keep Input.* and numeric Answer.* columns, drop everything else."""
    keep = []
    for col in df.columns:
        if col in ID_COLUMNS or col.startswith('Input.'):
            keep.append(col)
        elif col.startswith('Answer.') and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            keep.append(col)
    out = df[keep].copy()
    for col in ID_COLUMNS:
        if col in out:
            out[col] = out[col].map(lambda v: _hash(v, salt))
    return out


def check_no_ids(paths):
    for path in paths:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
        m = WORKER_ID.search(text)
        if m:
            raise ValueError(f'worker-ID-like string in {path} (at offset {m.start()})')


def _salt():
    salt = os.environ.get('TS_T5_SALT')
    if not salt:
        sys.exit('TS_T5_SALT is not set. Create ~/.config/ts_t5/salt and run with '
                 'TS_T5_SALT="$(cat ~/.config/ts_t5/salt)".')
    return salt


def _scrub_html(text):
    text = EMAIL.sub('[contact removed]', text)
    text = re.sub(r'contact [^()<]*\(email', 'contact the requester (email', text)
    return re.sub(r'Kim Cheng SHEANG, Horacio Saggion', '[requester names removed]', text)


def build(src, out):
    salt = _salt()
    res = src / 'results'
    final = res / '5tokens' / 'final data'
    out.mkdir(exist_ok=True)
    batches = [(final / '1. mTurks ratings' / f'Batch_v{i}.csv', f'5tokens_batch_v{i}.csv') for i in (1, 2, 3)]
    batches.append((res / '4tokens' / 'Batch_4442003_batch_results.csv', '4tokens_batch_4442003.csv'))
    (out / 'mturk_batches').mkdir(exist_ok=True)
    for path, name in batches:
        df = anonymise(pd.read_csv(path), salt)
        if name.startswith('4tokens'):  # a worker hash shared with the 5-token batches links to the public raw exports
            df = df.drop(columns='WorkerId')
        df.to_csv(out / 'mturk_batches' / name, index=False)
    for tok, d in (('5tokens', final / '2. extracted ratings'), ('4tokens', res / '4tokens' / 'final_results')):
        dest = out / 'extracted_ratings' / tok
        dest.mkdir(parents=True, exist_ok=True)
        for m in MODELS:
            pd.read_excel(d / f'{m}.xlsx', header=None).to_csv(dest / f'{m}.csv', index=False, header=False)
        shutil.copy(d / 'complex.txt', dest / 'complex.txt')
    hit = src / 'HIT_designs'
    for f in [*hit.glob('*.html'), *(hit / 'others').glob('*.html')]:  # not the third-party kriz/ designs
        dest = out / 'HIT_designs' / f.relative_to(hit)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_scrub_html(f.read_text(encoding='utf-8')), encoding='utf-8')
    check_no_ids([p for p in out.rglob('*') if p.is_file()])


def scores(out):
    """Mean rating per system and aspect, from the anonymised MTurk batches."""
    rows = []
    for study in ('5tokens', '4tokens'):
        files = sorted((out / 'mturk_batches').glob(f'{study}_*.csv'))
        sums = {(m, a): [] for m in range(4) for a in ASPECTS}
        for f in files:
            df = pd.read_csv(f)
            for j in range(1, 6):
                for k in range(1, 5):
                    mid = df[f'Input.simp_sent_id_{j}{k}']
                    for a in ASPECTS:
                        for m in range(4):
                            sums[(m, a)].extend(df.loc[mid == m, f'Answer.{a}_{j}{k}'].tolist())
        for m, name in enumerate(MODELS):
            rows.append({'study': study, 'system': name, 'n': len(sums[(m, 'fluency')]),
                         **{a: round(sum(sums[(m, a)]) / len(sums[(m, a)]), 3) for a in ASPECTS}})
    return pd.DataFrame(rows)


def extracted_scores(out, study):
    """Cross-check: means from the extracted-ratings CSVs (header row; per aspect 5 ratings + AVG)."""
    rows = []
    for m in MODELS:
        df = pd.read_csv(out / 'extracted_ratings' / study / f'{m}.csv', header=0)
        df = df[pd.to_numeric(df.iloc[:, 1], errors='coerce').notna()]  # drop trailing summary rows
        vals = {a: df.iloc[:, 1 + 6 * i: 6 + 6 * i].astype(float).values.mean() for i, a in enumerate(ASPECTS)}
        rows.append({'study': study, 'system': m, **{a: round(v, 3) for a, v in vals.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('src', type=Path)
    s = sub.add_parser('scores')
    for p in (b, s):
        p.add_argument('--out', type=Path, default=Path('human_evaluation'))
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.src, a.out)
    else:
        print('From mturk_batches/*.csv'); print(scores(a.out).to_string(index=False))
        for st in ('5tokens', '4tokens'):
            print(f'\nFrom extracted_ratings/{st}/*.csv'); print(extracted_scores(a.out, st).to_string(index=False))


if __name__ == '__main__':
    main()
