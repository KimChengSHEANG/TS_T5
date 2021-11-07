from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from easse.cli import evaluate_system_output
from source.model import T5FineTuner
from easse.report import get_all_scores

from itertools import product
import json
from source.preprocessor import Preprocessor
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from source.resources import get_data_filepath, get_last_experiment_dir, EXP_DIR, TURKCORPUS_DATASET, REPO_DIR
from source.helper import write_lines, yield_lines, count_line, read_lines, log_stdout, generate_hash
from easse.sari import corpus_sari
import time


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(12)
model = None
device = None
tokenizer = None
model_dir = None
_model_dirname = None

max_len = 256



def load_model(model_dirname=None):
    print("Load model", model_dirname)
    global model, tokenizer, device, model_dir, _model_dirname, max_len
    if model_dirname is None:  # default
        model_dir = get_last_experiment_dir()
    else:
        model_dir = EXP_DIR / model_dirname

    if _model_dirname is None or model_dirname != _model_dirname:
        print("loading model...")
        _model_dirname = model_dir.stem

        print("Model dir: ", model_dir)
        params_filepath = model_dir / "params.json"
        params = json.load(params_filepath.open('r'))
        max_len = int(params['max_seq_length'])
        # max_len = 80

        if (model_dir / 'pytorch_model.bin').exists():
            model = T5ForConditionalGeneration.from_pretrained(model_dir)
            tokenizer = T5TokenizerFast.from_pretrained(params['tokenizer_name_or_path'])
        else:
            checkpoints = list(model_dir.glob('checkpoint*'))
            best_checkpoint = sorted(checkpoints, reverse=True)[0] 
            print('check_point:', best_checkpoint)
            T5_model = T5FineTuner.load_from_checkpoint(checkpoint_path=best_checkpoint)
            model = T5_model.model
            tokenizer = T5_model.tokenizer    
            
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("device ", device)
        model = model.to(device)


def generate(sentence, preprocessor):
    # if not torch.cuda.is_available():
    #     print("Simplifying: ", sentence)
    sentence = preprocessor.encode_sentence(sentence)
    text = "simplify: " + sentence
    encoding = tokenizer(text, max_length=max_len,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_len,
        num_beams=8,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs


    

def evaluate(orig_filepath, sys_filepath, ref_filepaths):
    orig_sents = read_lines(orig_filepath)
    refs_sents = [read_lines(filepath) for filepath in ref_filepaths]
    # print(sys_filepath.name, f"Sari score:: ({sari_score})", )
    return corpus_sari(orig_sents, read_lines(sys_filepath), refs_sents)


def evaluate_all_metrics(orig_filepath, sys_filepath, ref_filepaths):
    orig_sents = read_lines(orig_filepath)
    refs_sents = [read_lines(filepath) for filepath in ref_filepaths]
    # return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=True)
    # return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=False)
    return get_all_scores(orig_sents, read_lines(sys_filepath), refs_sents, lowercase=True)


def evaluate_on(dataset, features_kwargs, phase, model_dirname=None):
    load_model(model_dirname)
    preprocessor = Preprocessor(features_kwargs)
    output_dir = model_dir / "outputs"
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        # log_params(output_dir / f"{features_hash}_features_kwargs.json", features_kwargs)
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'complex')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        # ref_filepaths = [get_data_filepath(dataset, phase, 'simple.turk', i) for i in range(8)]
        ref_filepath = get_data_filepath(dataset, phase, 'simple')
        print(pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, preprocessor)

        # print("Evaluate: ", pred_filepath)
        with log_stdout(output_score_filepath):
            # print("features_kwargs: ", features_kwargs)

            # refs = [[line] for line in yield_lines(ref_filepath)]
            # score = corpus_sari(read_lines(complex_filepath), read_lines(pred_filepath), refs)
            # print(len(read_lines(complex_filepath)), len(read_lines(pred_filepath)) )
            # print([len(s) for s in refs])

            # scores = get_all_scores(read_lines(complex_filepath), read_lines(pred_filepath), refs)
            score = evaluate(complex_filepath, pred_filepath, [ref_filepath])
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
            if "WordRatioFeature" in features_kwargs:
                print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            if "CharRatioFeature" in features_kwargs:
                print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            if "LevenshteinRatioFeature" in features_kwargs:
                print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            if "WordRankRatioFeature" in features_kwargs:
                print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            if "DependencyTreeDepthRatioFeature" in features_kwargs:
                print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("{:.2f} \t ".format(score))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


def simplify_file(complex_filepath, output_filepath, features_kwargs, model_dirname=None, post_processing=True):
    load_model(model_dirname)
    preprocessor = Preprocessor(features_kwargs)
    
    total_lines = count_line(complex_filepath)
    print(complex_filepath)
    print(complex_filepath.stem)

    output_file = Path(output_filepath).open("w")

    for n_line, complex_sent in enumerate(yield_lines(complex_filepath), start=1):
        output_sents = generate(complex_sent, preprocessor)
        print(f"{n_line+1}/{total_lines}", " : ", output_sents)
        if output_sents:
            output_file.write(output_sents[0] + "\n")
        else:
            output_file.write("\n")
    output_file.close()
    
    if post_processing: post_process(output_file)

def post_process(filepath):
    lines = []
    for line in yield_lines(filepath):
        lines.append(line.replace("''", '"'))
    write_lines(lines, filepath)
    
def evaluate_on_TurkCorpus(features_kwargs, phase, model_dirname=None):
    dataset = TURKCORPUS_DATASET

    model_dir = get_last_experiment_dir() if model_dirname is None else EXP_DIR / model_dirname
    output_dir = model_dir / "outputs"
    
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        # log_params(output_dir / f"{features_hash}_features_kwargs.json", features_kwargs)
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'complex')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        ref_filepaths = [get_data_filepath(dataset, phase, 'simple.turk', i) for i in range(8)]
        print(pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)
            
        # print("Evaluate: ", pred_filepath)
        with log_stdout(output_score_filepath):
            # print("features_kwargs: ", features_kwargs)
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
            scores = evaluate_system_output(test_set="turkcorpus_test", sys_sents_path=str(pred_filepath), lowercase=True)
            if "WordRatioFeature" in features_kwargs:
                print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            if "CharRatioFeature" in features_kwargs:
                print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            if "LevenshteinRatioFeature" in features_kwargs:
                print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            if "WordRankRatioFeature" in features_kwargs:
                print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            if "DependencyTreeDepthRatioFeature" in features_kwargs:
                print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))
            # print("{:.2f} \t {:.2f} \t {:.2f} ".format(scores['SARI'], scores['BLEU'], scores['FKGL']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
            return scores['sari']
            
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))


def evaluate_on_asset(features_kwargs, phase, model_dirname=None):
    dataset = "asset"
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    
    model_dir = get_last_experiment_dir() if model_dirname is None else EXP_DIR / model_dirname
    output_dir = model_dir / "outputs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir / f"score_{features_hash}_{dataset}_{phase}_log.txt"
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        start_time = time.time()
        complex_filepath = get_data_filepath(dataset, phase, 'orig')
        pred_filepath = output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        # ref_filepaths = [get_data_filepath(dataset, phase, 'simp', i) for i in range(10)]
        print(pred_filepath)
        if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
            print("File is already processed.")
        else:
            simplify_file(complex_filepath, pred_filepath, features_kwargs, model_dirname)
            
        with log_stdout(output_score_filepath):
            # scores = evaluate_all_metrics(complex_filepath, pred_filepath, ref_filepaths)
            scores = evaluate_system_output(test_set="asset_test", sys_sents_path=str(pred_filepath), lowercase=True)
            if "WordRatioFeature" in features_kwargs:
                print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
            if "CharRatioFeature" in features_kwargs:
                print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
            if "LevenshteinRatioFeature" in features_kwargs:
                print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
            if "WordRankRatioFeature" in features_kwargs:
                print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
            if "DependencyTreeDepthRatioFeature" in features_kwargs:
                print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
            print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))

            print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

