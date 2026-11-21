"""Repo paths and small file/hash helpers (ported from source/resources.py and source/helper.py)."""
import hashlib
from pathlib import Path

# ponytail: repo-anchored paths; editable install only, add a TS_T5_HOME env var if wheels are wanted
REPO_DIR = Path(__file__).resolve().parents[2]
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DUMPS_DIR = RESOURCES_DIR / 'DUMPS'
PROCESSED_DATA_DIR = RESOURCES_DIR / 'processed_data'
EXP_DIR = REPO_DIR / 'experiments'

WIKILARGE = 'wikilarge'
TURKCORPUS = 'turkcorpus'
ASSET = 'asset'


def require_resources():
    if not RESOURCES_DIR.is_dir():
        raise FileNotFoundError(
            f'{RESOURCES_DIR} not found. ts-t5 needs the repo checkout; '
            'clone it and run `pip install -e .` from the repo root.')


def get_data_filepath(dataset, phase, type, i=None):
    suffix = '' if i is None else f'.{i}'
    return DATASETS_DIR / dataset / f'{dataset}.{phase}.{type}{suffix}'


def yield_lines(p):
    with Path(p).open('r') as f:
        for line in f:
            yield line.rstrip()


def read_lines(p):
    return list(yield_lines(p))


def write_lines(lines, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def count_line(p):
    with Path(p).open('r') as f:
        return sum(1 for _ in f)


def generate_hash(data):
    return hashlib.md5(str(data).encode()).hexdigest()
