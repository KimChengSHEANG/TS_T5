"""Recompute the README numbers that are not in the paper or in docs/.

    python scripts/readme_analysis.py                        # data, outputs, SARI intervals, human ratings (about 1 min)
    python scripts/readme_analysis.py --format-check MODEL   # also the number-format experiment (about 1 hour)

Uses committed files only, plus MODEL for --format-check.
"""
import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from easse import sari
from easse.cli import get_orig_and_refs_sents
from easse.utils.preprocessing import normalize
from scipy import stats  # installed with easse

from ts_t5.data import REPO_DIR, get_data_filepath, read_lines, write_lines
from ts_t5.evaluate import EXPECTED, score_file

BOOTSTRAP, SEED = 10_000, 12
MODELS = ('YATS', 'DMASS-DCSS', 'MUSS', 'TS-T5')  # index = Input.simp_sent_id_*
ASPECTS = ('fluency', 'simplicity', 'adequacy')
OUT = REPO_DIR / 'system_outputs'
OURS = {'4tokens': 'TS_T5_4tokens/d859b9c4aab9f7a9c719167728399278',
        '5tokens': 'TS_T5_5tokens/267a9c5912b4c0e37bf8bf7260f7576d'}
BASELINES = {'MUSS': 'MUSS(Bart+ACCESS).txt', 'ACCESS': 'ACCESS.txt', 'DMASS-DCSS': 'DMASS-DCSS.txt',
             'Dress-Ls': 'Dress-Ls.txt', 'YATS': 'yats.txt'}


def key(text):
    """Letters and digits only, so tokenisation and case do not matter when matching sentences."""
    return re.sub(r'[\W_]+', '', str(text).lower())


def test_sources(test_set):
    return read_lines(get_data_filepath(test_set, 'test', 'orig' if test_set == 'asset' else 'complex'))


def interval(samples):
    return '[%.2f, %.2f]' % tuple(np.percentile(samples, [2.5, 97.5]))


def data_report():
    src = read_lines(get_data_filepath('wikilarge', 'train', 'complex'))
    train = set(map(key, src))
    print(f'WikiLarge train: {len(src)} pairs, {len(set(src))} distinct sources')
    for name, phase in (('turkcorpus', 'test'), ('turkcorpus', 'valid'), ('wikilarge', 'valid')):
        rows = read_lines(get_data_filepath(name, phase, 'complex'))
        print(f'{name} {phase}: {sum(key(r) in train for r in rows)}/{len(rows)} sources also in WikiLarge train')
    same = sum(key(a) == key(b) for a, b in zip(test_sources('turkcorpus'), test_sources('asset')))
    print(f'TurkCorpus and ASSET test share {same} sources')
    return train


def sari_report(test_set, train):
    raw = test_sources(test_set)
    # easse's bundled references, as in score_file. The files in resources/ are tokenised differently.
    orig, refs = get_orig_and_refs_sents(f'{test_set}_test')
    norm = lambda sents: [normalize(s, True, '13a') for s in sents]
    orig, refs = norm(orig), [norm(r) for r in refs]
    files = {m: OUT / f'{stem}_{test_set}.test.txt' for m, stem in OURS.items()}
    if test_set == 'turkcorpus':
        files.update({m: OUT / f for m, f in BASELINES.items()})
    weights = np.random.default_rng(SEED).multinomial(len(orig), [1 / len(orig)] * len(orig), size=BOOTSTRAP)
    unseen = np.array([key(s) not in train for s in raw]).astype(int)

    def operations(table, w):
        totals = (w @ table.reshape(len(orig), -1)).reshape(len(w), 9, 4).astype(float)
        return np.array([(200 * totals[:, i] / np.maximum(totals[:, i + 1] + totals[:, i + 2], 1)).mean(1)
                         for i in (0, 3, 6)])

    point, boot = {}, {}
    for model, path in files.items():
        lines = read_lines(path)
        out = norm(lines)
        table = np.array([sari.compute_ngram_stats([orig[i]], [out[i]], [[r[i]] for r in refs])
                          for i in range(len(orig))])
        ops = operations(table, np.ones((1, len(orig)), int))[:, 0]
        point[model], boot[model] = ops.mean(), operations(table, weights).mean(0)
        if model in EXPECTED:
            assert abs(point[model] - EXPECTED[model][test_set]['sari']) < 0.006, (model, test_set, point[model])
        print(f'{test_set} {model}: SARI {point[model]:.2f} {interval(boot[model])} add/keep/delete {ops.round(2)} '
              f'unseen-only {operations(table, unseen[None]).mean():.2f} '
              f'copies {sum(key(a) == key(b) for a, b in zip(lines, raw))} '
              f'length ratio {np.median([len(a) / len(b) for a, b in zip(lines, raw)]):.3f}')
    for a in OURS:
        for b in files:
            if a != b:
                print(f'  {a} minus {b}: {point[a] - point[b]:+.2f} {interval(boot[a] - boot[b])}')


def human_report():
    index = {key(s): i for i, s in enumerate(test_sources('turkcorpus'))}
    for study in OURS:
        rows = []
        for path in sorted((REPO_DIR / 'human_evaluation' / 'mturk_batches').glob(f'{study}_*.csv')):
            for _, r in pd.read_csv(path).iterrows():
                for j in range(1, 6):
                    for k in range(1, 5):
                        rows.append([key(r[f'Input.orig_sent_{j}']), r[f'Input.simp_sent_id_{j}{k}'],
                                     r[f'Input.simp_sent_{j}{k}'], *(r[f'Answer.{a}_{j}{k}'] for a in ASPECTS)])
        df = pd.DataFrame(rows, columns=['src', 'model', 'text', *ASPECTS])
        assert len(df) == 2000, (study, len(df))
        rated = df[df.model == 3].drop_duplicates('src')
        released = read_lines(OUT / f'{OURS[study]}_turkcorpus.test.txt')
        print(f'{study}: {sum(key(r.text) == key(released[index[r.src]]) for r in rated.itertuples())}/100 rated '
              'TS-T5 sentences equal the released default-token output')
        for aspect in ASPECTS:
            means = df.pivot_table(index='src', columns='model', values=aspect, aggfunc='mean')
            for m in range(3):
                d = (means[3] - means[m]).to_numpy()
                b = np.random.default_rng(SEED).choice(d, (BOOTSTRAP, len(d))).mean(1)
                print(f'  {aspect} TS-T5 minus {MODELS[m]}: {d.mean():+.3f} {interval(b)} '
                      f'paired t p={stats.ttest_rel(means[3], means[m]).pvalue:.2g}')


def format_report(model):
    """Training writes C_0.80 and inference writes C_0.8. Same number, different T5 tokens."""
    import torch

    from ts_t5.simplify import Simplifier
    s = Simplifier(model)
    src = test_sources('turkcorpus')
    with tempfile.TemporaryDirectory() as tmp:
        for value in ('0.5', '0.50', '0.8', '0.80', '0.9', '0.90', '1.0', '1.00'):
            out = []
            for x in src:
                enc = s.tokenizer(f'simplify: C_{value} L_0.75 WR_0.75 DTD_0.75  {x}', max_length=s.max_len,
                                  truncation=True, return_tensors='pt').to(s.device)
                with torch.no_grad():
                    ids = s.model.generate(**enc, num_beams=8, do_sample=False, early_stopping=True, max_length=s.max_len)
                out.append(s.tokenizer.decode(ids[0], skip_special_tokens=True).replace("''", '"'))
            path = Path(tmp) / f'{value}.txt'
            write_lines(out, path)
            print(f'C_{value}: median length ratio {np.median([len(o) / len(x) for o, x in zip(out, src)]):.3f} '
                  f'{score_file("turkcorpus", path)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--format-check', metavar='MODEL', help='a 4-token model id or directory')
    args = p.parse_args()
    train = data_report()
    for test_set in ('turkcorpus', 'asset'):
        sari_report(test_set, train)
    human_report()
    if args.format_check:
        format_report(args.format_check)


if __name__ == '__main__':
    main()
