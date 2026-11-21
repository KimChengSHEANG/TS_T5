"""Control-token features (ported from source/preprocessor.py).

Token order in the feature dict is token order in the encoded line.
"""
import multiprocessing
import pickle
import re
import shutil
import time
from functools import lru_cache
from multiprocessing import Pool
from string import punctuation

import Levenshtein
import numpy as np

from ts_t5.data import (DUMPS_DIR, PROCESSED_DATA_DIR, count_line, generate_hash,
                        get_data_filepath, write_lines)

PHASES = ('train', 'valid', 'test')
SPACY_MODEL = 'en_core_web_sm'

FEATURES_4TOKENS = {
    'CharRatioFeature': {'target_ratio': 0.95},
    'LevenshteinRatioFeature': {'target_ratio': 0.75},
    'WordRankRatioFeature': {'target_ratio': 0.75},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75},
}
FEATURES_5TOKENS = {'WordRatioFeature': {'target_ratio': 1.05}, **FEATURES_4TOKENS}


def round(val):
    return '%.2f' % val


def safe_division(a, b):
    return a / b if b else 0


@lru_cache(maxsize=1)
def get_tokenizer():
    from sacremoses import MosesTokenizer
    return MosesTokenizer(lang='en')


def tokenize(sentence):
    return get_tokenizer().tokenize(sentence)


@lru_cache(maxsize=1)
def get_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))


@lru_cache(maxsize=1)
def get_spacy_model():
    import spacy
    if not spacy.util.is_package(SPACY_MODEL):
        spacy.cli.download(SPACY_MODEL)
    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=1)
def get_word2rank():
    path = DUMPS_DIR / 'glove.42B.300d.pk'
    with path.open('rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1024)
def is_punctuation(word):
    return ''.join([char for char in word if char not in punctuation]) == ''


@lru_cache(maxsize=128)
def remove_punctuation(text):
    return ' '.join([word for word in tokenize(text) if not is_punctuation(word)])


def remove_stopwords(text):
    stop = get_stopwords()
    return ' '.join([w for w in tokenize(text) if w.lower() not in stop])


class RatioFeature:
    def __init__(self, feature_extractor, target_ratio=0.8):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio

    def encode_sentence(self, sentence):
        return f'{self.name}_{self.target_ratio}'

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        return f'{self.name}_{self.feature_extractor(complex_sentence, simple_sentence)}', simple_sentence

    def decode_sentence(self, encoded_sentence):
        return encoded_sentence

    @property
    def name(self):
        class_name = self.__class__.__name__.replace('RatioFeature', '')
        name = ''
        for word in re.findall('[A-Z][^A-Z]*', class_name):
            if word:
                name += word[0]
        if not name:
            name = class_name
        return name


class WordRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_length_ratio, *args, **kwargs)

    def get_word_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(tokenize(simple_sentence)), len(tokenize(complex_sentence))))


class CharRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_char_length_ratio, *args, **kwargs)

    def get_char_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(simple_sentence), len(complex_sentence)))


class LevenshteinRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_levenshtein_ratio, *args, **kwargs)

    def get_levenshtein_ratio(self, complex_sentence, simple_sentence):
        return round(Levenshtein.ratio(complex_sentence, simple_sentence))


class WordRankRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_rank_ratio, *args, **kwargs)

    def get_word_rank_ratio(self, complex_sentence, simple_sentence):
        return round(min(safe_division(self.get_lexical_complexity_score(simple_sentence),
                                       self.get_lexical_complexity_score(complex_sentence)), 2))

    def get_lexical_complexity_score(self, sentence):
        words = tokenize(remove_stopwords(remove_punctuation(sentence)))
        words = [word for word in words if word in get_word2rank()]
        if len(words) == 0:
            return np.log(1 + len(get_word2rank()))
        return np.quantile([self.get_rank(word) for word in words], 0.75)

    @lru_cache(maxsize=5000)
    def get_rank(self, word):
        rank = get_word2rank().get(word, len(get_word2rank()))
        return np.log(1 + rank)


class DependencyTreeDepthRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_dependency_tree_depth_ratio, *args, **kwargs)

    def get_dependency_tree_depth_ratio(self, complex_sentence, simple_sentence):
        return round(
            safe_division(self.get_dependency_tree_depth(simple_sentence),
                          self.get_dependency_tree_depth(complex_sentence)))

    @lru_cache(maxsize=1024)
    def get_dependency_tree_depth(self, sentence):
        def get_subtree_depth(node):
            if len(list(node.children)) == 0:
                return 0
            return 1 + max([get_subtree_depth(child) for child in node.children])

        tree_depths = [get_subtree_depth(s.root) for s in self.spacy_process(sentence).sents]
        if len(tree_depths) == 0:
            return 0
        return max(tree_depths)

    @lru_cache(maxsize=10 ** 6)
    def spacy_process(self, text):
        return get_spacy_model()(text)


class Preprocessor:
    def __init__(self, features_kwargs=None):
        self.features = self.get_features(features_kwargs or {})
        if features_kwargs:
            # hash of the bytes repr: names the processed_data dirs, do not change
            self.hash = generate_hash(str(features_kwargs).encode())
        else:
            self.hash = 'no_feature'

    def get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs)

    def get_features(self, feature_kwargs):
        return [self.get_class(name, **kwargs) for name, kwargs in feature_kwargs.items()]

    def encode_sentence(self, sentence):
        if self.features:
            line = ''
            for feature in self.features:
                line += feature.encode_sentence(sentence) + ' '
            line += ' ' + sentence
            return line.rstrip()
        return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        if self.features:
            line = ''
            for feature in self.features:
                processed_complex, _ = feature.encode_sentence_pair(complex_sentence, simple_sentence)
                line += processed_complex + ' '
            line += ' ' + complex_sentence
            return line.rstrip()
        return complex_sentence

    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def pool_encode_sentence_pair(self, args):
        complex_sent, simple_sent, queue = args
        queue.put(1)
        return self.encode_sentence_pair(complex_sent, simple_sent)

    def encode_file_pair(self, complex_filepath, simple_filepath):
        line_count = count_line(simple_filepath)
        with open(complex_filepath) as fc, open(simple_filepath) as fs:
            pairs = [(c.rstrip(), s.rstrip()) for c, s in zip(fc, fs)]
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        pool = Pool(processes=multiprocessing.cpu_count())
        res = pool.map_async(self.pool_encode_sentence_pair, [(c, s, queue) for c, s in pairs])
        while not res.ready():
            print(f'Preprocessing: {queue.qsize()} / {line_count}')
            time.sleep(0.5)
        encoded_sentences = res.get()
        pool.close()
        pool.join()
        return encoded_sentences

    def preprocess_dataset(self, dataset):
        out_dir = PROCESSED_DATA_DIR / self.hash / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Preprocessing dataset: {dataset}')
        for phase in PHASES:
            complex_filepath = get_data_filepath(dataset, phase, 'complex')
            simple_filepath = get_data_filepath(dataset, phase, 'simple')
            complex_out = out_dir / complex_filepath.name
            simple_out = out_dir / simple_filepath.name
            if complex_out.exists() and simple_out.exists():
                continue
            print(f'Preprocessing files: {complex_filepath.name} {simple_filepath.name}')
            write_lines(self.encode_file_pair(complex_filepath, simple_filepath), complex_out)
            shutil.copy(simple_filepath, simple_out)
        print(f'Preprocessing dataset "{dataset}" is finished.')
        return out_dir
