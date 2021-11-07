# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.resources import DATASETS_DIR, REPO_DIR
from source.download_data import download_5tokens_trained_model
from source.evaluate import simplify_file


features_kwargs = {
    'CharRatioFeature': {'target_ratio': 0.95},
    'LevenshteinRatioFeature': {'target_ratio': 0.75},
    'WordRankRatioFeature': {'target_ratio': 0.75},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
}

# Download the pretrained model from gdrive if not exists
download_5tokens_trained_model()

# TurkCorpus
# input filepath
turkcorpus_complex_filepath = DATASETS_DIR / 'turkcorpus/turkcorpus.test.complex' 

# output filepath
output_dir = REPO_DIR / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
output_filepath = output_dir / f'{turkcorpus_complex_filepath.stem}.txt' 

simplify_file(turkcorpus_complex_filepath, output_filepath, features_kwargs, 'model_5tokens')


