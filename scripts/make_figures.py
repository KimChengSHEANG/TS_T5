"""Redraw the 2021 compression-ratio histograms as PNG (port of the plotly scripts in D8 plots/).

Needs matplotlib (not a runtime dependency) and the 2021 system outputs on valid, which are not in the repo:

    python scripts/make_figures.py --data /Volumes/D8/workspace/TS_T5/plots/data --out docs/figures

Ratios: char ratio = len(simple) / len(complex); Levenshtein similarity = Levenshtein.ratio(complex, simple).
"""
import argparse
from pathlib import Path

import Levenshtein
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from ts_t5.data import get_data_filepath, read_lines  # noqa: E402

C_VALUES = ['0.25', '0.5', '0.75', '1.0']
# md5 of str(features_kwargs): C_x = {C: x}; W_x = {C: 1.0, W: x} (see docs/ABLATIONS.md for the hash method)
W_HASHES = {'0.25': '422d40a346a5fc798a3ffca5bfd4cab1', '0.5': 'c81082bc173e982d488d09ce2af089ff',
            '0.75': '2327886580d32381e6e47f3d3b1fc990', '1.0': '39923bc591d6d12f3d346530e6f62b89'}


def char_ratio(c, s):
    return len(s) / len(c) if c else 0


def lev_sim(c, s):
    return Levenshtein.ratio(c, s)


def hist(ax, series, fn, title, xlabel, xlim=(0.0, 1.5), bins=60):
    for label, (orig, sys_, color) in series.items():
        ax.hist([fn(o, s) for o, s in zip(orig, sys_)], bins=bins, range=xlim, alpha=0.55, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)


def fig_w(data, out, orig, ref):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, v in zip(axes, C_VALUES):
        sys_ = read_lines(data / f'W_{v}_{W_HASHES[v]}_turkcorpus.valid.txt')
        hist(ax, {f'C=1.0, W={v}': (orig, sys_, '#1565c0'), 'Reference': (orig, ref, '#228B22')},
             char_ratio, f'W = {v}', 'Char ratio')
    fig.tight_layout()
    fig.savefig(out / 'char_ratio_by_W.png', dpi=130)


def fig_c(data, out, orig, ref):
    for name, fn, label in (('char_ratio', char_ratio, 'Char ratio'), ('levenshtein', lev_sim, 'Levenshtein similarity')):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
        for ax, v in zip(axes, C_VALUES):
            s4 = read_lines(data / '4tokens' / f'C_{v}.txt')
            s5 = read_lines(data / '5tokens' / f'C_{v}.txt')
            hist(ax, {f'4-token, C={v}': (orig, s4, '#ef5350'), f'5-token, W=1.0, C={v}': (orig, s5, '#1565c0'),
                      'Reference': (orig, ref, '#228B22')}, fn, f'C = {v}', label, xlim=(0.25, 1.25), bins=50)
        fig.tight_layout()
        fig.savefig(out / f'{name}_4tok_vs_5tok.png', dpi=130)


def fig_dataset(out):
    wl = (read_lines(get_data_filepath('wikilarge', 'train', 'complex')),
          read_lines(get_data_filepath('wikilarge', 'train', 'simple')))
    tc = (read_lines(get_data_filepath('turkcorpus', 'valid', 'complex')) + read_lines(get_data_filepath('turkcorpus', 'test', 'complex')),
          read_lines(get_data_filepath('turkcorpus', 'valid', 'simple.turk', 0)) + read_lines(get_data_filepath('turkcorpus', 'test', 'simple.turk', 0)))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (c, s), color in (('WikiLarge train', wl, '#ef5350'), ('TurkCorpus valid+test (ref 0)', tc, '#1565c0')):
        ax.hist([char_ratio(a, b) for a, b in zip(c, s)], bins=80, range=(0, 2.5), alpha=0.55, label=label, color=color, density=True)
    ax.set_xlabel('Char ratio (simple / complex)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'dataset_char_ratio.png', dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path, help='D8 plots/data directory')
    ap.add_argument('--out', default=Path('docs/figures'), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    orig = read_lines(get_data_filepath('turkcorpus', 'valid', 'complex'))
    ref = read_lines(get_data_filepath('turkcorpus', 'valid', 'simple.turk', 1))
    fig_w(a.data, a.out, orig, ref)
    fig_c(a.data, a.out, orig, ref)
    fig_dataset(a.out)


if __name__ == '__main__':
    main()
