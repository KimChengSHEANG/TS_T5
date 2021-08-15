# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.helper import get_max_seq_length
from source.train import run_training
from source.resources import get_experiment_dir, WIKILARGE_DATASET
import torch

dataset = WIKILARGE_DATASET
args_dict = dict(
    model_name='t5-base',
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_epsilon=1e-8,
    warmup_steps=5,
    train_batch_size=6,
    valid_batch_size=6,
    num_train_epochs=5,
    custom_loss=False,   
    gradient_accumulation_steps=1, #16
    n_gpu=torch.cuda.device_count(),
    # early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=12,
    nb_sanity_val_steps=0,
    train_sample_size=1, # 1 = 100%, 0.5 = 50%
    valid_sample_size=1, 
)

features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
}
args_dict['features_kwargs'] = features_kwargs
run_training(args_dict, dataset)
