import pytest

from ts_t5 import data
from ts_t5.data import generate_hash, get_data_filepath, read_lines

F4 = {'CharRatioFeature': {'target_ratio': 0.95}, 'LevenshteinRatioFeature': {'target_ratio': 0.75},
      'WordRankRatioFeature': {'target_ratio': 0.75}, 'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}}
F5 = {'WordRatioFeature': {'target_ratio': 1.05}, **F4}


def test_hash_matches_published_outputs():
    assert generate_hash(F4) == 'd859b9c4aab9f7a9c719167728399278'
    assert generate_hash(F5) == '267a9c5912b4c0e37bf8bf7260f7576d'


def test_test_sets_present():
    assert len(read_lines(get_data_filepath('turkcorpus', 'test', 'complex'))) == 359
    assert len(read_lines(get_data_filepath('asset', 'test', 'orig'))) == 359  # no trailing newline: wc says 358


def test_missing_resources_message(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'RESOURCES_DIR', tmp_path / 'nope')
    with pytest.raises(FileNotFoundError, match='pip install -e .'):
        data.require_resources()
