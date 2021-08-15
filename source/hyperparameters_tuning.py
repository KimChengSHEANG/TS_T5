# -- fix path --
from pathlib import Path;
import sys;
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path
# -- end fix path --

from source.helper import get_max_seq_length, log_stdout
from source.train import run_train_tuning
from source.resources import EXP_DIR, WIKILARGE_DATASET
import torch
import optuna


def run_tuning(trial, params):
    dataset = WIKILARGE_DATASET
    args_dict = dict(
        model_name='t5-small',
        max_seq_length=get_max_seq_length(dataset),
        learning_rate=params['learning_rate'],
        weight_decay=0.1,
        adam_epsilon=params['adam_epsilon'],  # 1e-8,
        warmup_steps=5,
        train_batch_size=params['batch_size'],
        eval_batch_size=params['batch_size'],
        num_train_epochs=params['num_epochs'],
        gradient_accumulation_steps=16,
        n_gpu=torch.cuda.device_count(),
        early_stop_callback=False,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=12,
        nb_sanity_val_steps=0,
        train_sample_size=0.4,  # 0.3 = 30% , 1 = 100%
        valid_sample_size=0.2, # 0.2 = 20%
    )

    features_kwargs = {
        'WordRatioFeature': {'target_ratio': 0.8},
        'CharRatioFeature': {'target_ratio': 0.8},
        'LevenshteinRatioFeature': {'target_ratio': 0.8},
        'WordRankRatioFeature': {'target_ratio': 0.8},
        'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    args_dict['features_kwargs'] = features_kwargs
    return run_train_tuning(args_dict, dataset)


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        # 'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'batch_size': trial.suggest_categorical('batch_size', [6, 12, 18, 32]),
        'num_epochs': trial.suggest_categorical('num_epochs', [3, 5, 8]),
        # "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6),

        # "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),

    }
    return run_tuning(trial, params)


if __name__ == '__main__':

    # pruner: optuna.pruners.BasePruner = (
    #     optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    # )
    tuning_log_dir = EXP_DIR / 'tuning_logs'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    while tuning_logs.exists():
        i += 1
        tuning_logs = tuning_log_dir / f'logs_{i}.txt'

    with log_stdout(tuning_logs):
        # study = optuna.create_study(study_name='TS_T5_study', direction="minimize", storage='sqlite:///TS_T5_study.db')
        study = optuna.create_study(study_name='TS_T5_study', direction="minimize",
                                    storage=f'sqlite:///{tuning_log_dir}/TS_T5_study.db', load_if_exists=True)
        study.optimize(objective, n_trials=100)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
