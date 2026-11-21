"""Convert a Lightning checkpoint from ts_t5.train to a HF model directory.

    python scripts/convert_ckpt_to_hf.py experiments/exp_X/checkpoint-epoch=4.ckpt out/hf
    (reads params.json next to the checkpoint unless --params is given)
"""
import argparse
import json
from pathlib import Path

from ts_t5.train import convert

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=Path)
parser.add_argument('out_dir', type=Path)
parser.add_argument('--params', type=Path, help='params.json (default: next to the checkpoint)')
args = parser.parse_args()
params = json.loads((args.params or args.ckpt.parent / 'params.json').read_text())
print(convert(args.ckpt, args.out_dir, params))
