from itertools import product
from pathlib import Path
import time
import tempfile
import glob

from source.utils import download_url, unzip

REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
EXP_DIR = REPO_DIR / 'experiments'
DATASETS_DIR = RESOURCES_DIR / 'datasets'

PROCESSED_DATA_DIR = RESOURCES_DIR / "processed_data"

DUMPS_DIR = RESOURCES_DIR / "DUMPS"
TEMP_DIR = RESOURCES_DIR / "TEMP"

TURKCORPUS_DATASET = "turkcorpus"
WIKILARGE_DATASET = 'wikilarge'
ASSET_DATASET = "asset"
WIKILARGE_WIKIAUTO_DATASET = "wikilarge_wikiauto"
NEWSELA_DATASET = "newsela"

WORD_EMBEDDINGS_NAME = "glove.42B.300d"
WORD_FREQUENCY_FILEPATH = RESOURCES_DIR / 'others/enwiki_freq.txt'


LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']

# def get_processed_file(dataset, phase, type):
#     filename = f"{dataset}.{phase}.{type}"
#     return DATA_DIR / filename


def get_data_filepath(dataset, phase, type, i=None):
    suffix = ''
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{type}{suffix}'
    return DATASETS_DIR / dataset / filename

def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir == True: path.mkdir(parents=True, exist_ok=True)
    return path
def get_tuning_log_dir():
    log_dir = EXP_DIR / 'tuning_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    # i = 1
    # tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    # while(tuning_logs.exists()):
    #     i += 1
    #     tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    return log_dir
def get_last_experiment_dir():
    return sorted(list(EXP_DIR.glob('exp_*')), reverse=True)[0]

def download_glove(model_name, dest_dir):
    url = ''
    if model_name == 'glove.6B':
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    elif model_name == 'glove.42B.300d':
        url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    elif model_name == 'glove.840B.300d':
        url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    elif model_name == 'glove.twitter.27B':
        url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    else:
        possible_values = ['glove.6B', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B']
        raise ValueError('Unknown model_name. Possible values are {}'.format(possible_values))
    file_path = download_url(url, dest_dir)
    out_filepath = Path(file_path)
    out_filepath = out_filepath.parent / f'{out_filepath.stem}.txt'
    # print(out_filepath, out_filepath.exists())
    if not out_filepath.exists():
        print("Extracting: ", Path(file_path).name)
        unzip(file_path, dest_dir)

if __name__ == '__main__':
    # print(REPO_DIR)
    # print(get_experiments_dir())
    # print(str(Path(__file__).resolve().parent.parent))
    print(get_temp_filepath())