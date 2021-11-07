# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.evaluate import evaluate_on_TurkCorpus, evaluate_on_asset
from source.download_data import download_4tokens_trained_model


# Download the pretrained model from gdrive if not exists
download_4tokens_trained_model()


features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 1.05},
    'CharRatioFeature': {'target_ratio': 0.95},
    'LevenshteinRatioFeature': {'target_ratio': 0.75},
    'WordRankRatioFeature': {'target_ratio': 0.75},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
}

# generate and evaluate both datasets, the outputs and scores are stored in `experiments/model_5tokens/outputs`
evaluate_on_TurkCorpus(features_kwargs, 'test', 'model_4tokens')
evaluate_on_asset(features_kwargs, 'test', 'model_4tokens')