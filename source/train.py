from pathlib import Path;import sys;sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path
import time
from source.helper import log_params, log_stdout
from source.model import train
from source.evaluate import evaluate_on_TurkCorpus
from source.resources import get_experiment_dir, WIKILARGE_DATASET, get_last_experiment_dir, EXP_DIR
from source.preprocessor import Preprocessor
from optparse import OptionParser


def run_training(args_dict, dataset=WIKILARGE_DATASET):
    parser = OptionParser()
    parser.add_option("-r", "--resume",
                      action="store_true", dest="resume", default=False,
                      help="Resume from the previous training.")
    (options, args) = parser.parse_args()

    if options.resume:
        last_training_dir = get_last_experiment_dir()
        # last_training_dir = EXP_DIR / 'exp_1619204109284287'
        print("Resume from previous training: ", last_training_dir)
        args_dict['output_dir'] = last_training_dir
        args_dict['model_name_or_path'] = last_training_dir  # / 'checkpointepoch=2.ckpt'
    else:
        args_dict['output_dir'] = get_experiment_dir(create_dir=True)
        log_params(args_dict["output_dir"] / "params.json", args_dict)
    # args_dict['logging_dir'] = args_dict['output_dir'] / 'logs'

    preprocessor = Preprocessor(args_dict['features_kwargs'])
    preprocessor.preprocess_dataset(dataset)
    args_dict["dataset"] = dataset
    with log_stdout(args_dict['output_dir'] / "logs.txt"):
        train(args_dict)


def run_train_tuning(trial, args_dict, dataset=WIKILARGE_DATASET):
    dir_name = f'{int(time.time() * 1000000)}'
    exp_dir_path = EXP_DIR / f'tuning_experiments/exp_{dir_name}'
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    args_dict['output_dir'] = exp_dir_path
    log_params(args_dict["output_dir"] / "params.json", args_dict)

    preprocessor = Preprocessor(args_dict['features_kwargs'])
    preprocessor.preprocess_dataset(dataset)
    args_dict["dataset"] = dataset
    with log_stdout(args_dict['output_dir'] / "logs.txt"):
        run_training(trial, args_dict)
        return evaluate_on_TurkCorpus(args_dict['features_kwargs'], 'valid', exp_dir_path)
