# TS_T5 Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port TS_T5 to a current Python stack as the installable package `ts_t5`, reproduce the paper scores, publish both models on Hugging Face, and document the results and the human evaluation.

**Architecture:** The code moves from `source/` to `src/ts_t5/`, in one module per concern, behind an argparse CLI `ts-t5`. Inference behaviour (control-token string, decoding, post-filter) is ported line for line. Golden tests pin it against the hashes and outputs already in `system_outputs/`. The model is only uploaded after the gate passes: both models are scored with the new code and must match the README table.

**Tech Stack:** Python 3.11, PyTorch 2.x, transformers 4.x, Lightning 2.x, spaCy 3.x + `en_core_web_sm`, nltk, python-Levenshtein, easse, optuna (extra), huggingface_hub, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-20-ts-t5-release-design.md`

## Global Constraints

- Licence: CC BY-NC 4.0 for code and models (`LICENSE`, card `license: cc-by-nc-4.0`).
- HF repos: `kcsheang/ts-t5-base-4tokens`, `kcsheang/ts-t5-base-5tokens`, plus one Collection. Created **private** first.
- Decoding is fixed: `num_beams=8`, `do_sample=False`, `early_stopping=True`, `num_return_sequences=1`, `max_length` = `max_seq_length` (80 for both paper models), input padded to `max_length`.
- Model input: `"simplify: " + Preprocessor.encode_sentence(sentence)`. Token values are formatted `'%.2f'` only for pair encoding (training); at inference the value is `str(target_ratio)` as given (`C_0.95`, not `C_0.950`).
- Default tokens. 4-token: `C 0.95, L 0.75, WR 0.75, DTD 0.75`. 5-token: `W 1.05` followed by the same four. Dict order is token order.
- Gate: every SARI within ±0.1 of the README, BLEU and FKGL within ±0.5. Any miss means stop and report, with no upload.
- README targets: 4-token TurkCorpus 43.50/67.26/6.17, ASSET 45.05/72.24/6.33. 5-token TurkCorpus 43.34/65.54/5.55, ASSET 45.26/72.19/5.92.
- Evaluation: `easse` `evaluate_system_output(test_set="<name>_test", lowercase=True)`. If current easse moves scores, pin commit `8df369f025a7c927425482e802fb546fd5d06d84` and note why in `REPRODUCE.md`.
- `/Volumes/D8` is read-only: copy, never move or modify.
- Every outward step needs the author's explicit OK in the same session: push, HF upload, making repos public, merging to `main`.
- No new worker ID in any committed file: grep pattern `A[0-9A-Z]{12,}`. The existing `Human evaluation/results/raw from mTurk/*` files stay untouched.
- No Claude attribution in commits or PRs.

## Review Focus

1. **Sentence the post-filter rejects.** `generate` drops an output equal (case-insensitive) to the *encoded* input, then writes an empty line. The output file must keep one line per input, or easse misaligns. Test: `test_simplify_file_keeps_line_count` (Task 4).
2. **5-token conversion silently missing weights.** A Lightning state dict with an unexpected prefix loads as a random-init model and scores near zero. Conversion must fail on any missing or unexpected key except `lm_head.weight` (tied). Test: `test_convert_rejects_bad_state_dict` (Task 5).
3. **`features_kwargs` order.** Python dicts keep insertion order, but a JSON config or CLI parsing that reorders keys changes the token string and the output-file hash. Test: `test_cli_tokens_keep_order` (Task 7).
4. **spaCy model drift at training time.** A newer `en_core_web_sm` can change DTD ratios in training inputs. Inference is not affected, because the target ratio is literal. Test: `test_encode_pair_matches_2021_processed_data` (Task 2); a mismatch is reported in `REPRODUCE.md`, not hidden.
5. **Running from a non-repo directory.** Paths are anchored to the repo root, so a non-editable install has no `resources/`. It must fail with a clear message naming `pip install -e .`. Test: `test_missing_resources_message` (Task 1).

---

### Task 0: Branch and clean the working tree

**Files:**
- Modify: `source/model.py:40` and `source/helper.py` (revert)
- Add: `docs/superpowers/specs/2026-11-20-ts-t5-release-design.md`, this plan

- [ ] **Step 1:** `git switch -c release/hf-models`
- [ ] **Step 2:** `git restore source/model.py source/helper.py`. Check: `git diff --stat` is empty. Grep `source/` for non-ASCII bold glyphs; expect nothing: `grep -rnP '[^\x00-\x7F]' source/ | grep -v '#'`.
- [ ] **Step 3:** Add `.vscode/` to `.gitignore`. Commit the spec and plan: `git commit -m "docs: release design and plan"`.

### Task 1: Package skeleton, paths and I/O helpers

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `src/ts_t5/__init__.py`, `src/ts_t5/data.py`, `tests/test_data.py`, `.python-version` (`3.11`)

**Interfaces:**
- Produces in `ts_t5.data`:
  - Constants `REPO_DIR`, `RESOURCES_DIR`, `DATASETS_DIR`, `DUMPS_DIR`, `PROCESSED_DATA_DIR`, `EXP_DIR`, and dataset names `WIKILARGE`, `TURKCORPUS`, `ASSET`.
  - `get_data_filepath(dataset: str, phase: str, type: str, i: int | None = None) -> Path`
  - `read_lines(p) -> list[str]`, `yield_lines(p)`, `write_lines(lines, p)`, `count_line(p) -> int`
  - `generate_hash(data) -> str`: md5 of `str(data)`, unchanged.
  - `require_resources() -> None`: raises `FileNotFoundError` with a message containing `pip install -e .`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2:** Create the venv with `uv venv -p 3.11 && uv pip install -e '.[dev]'`. Run `pytest tests/test_data.py`. Expected: import errors.
- [ ] **Step 3:** Write `pyproject.toml`:
  - Package metadata: name `ts-t5`, `requires-python = ">=3.10"`, src layout, `[project.scripts] ts-t5 = "ts_t5.cli:main"`.
  - Pin the latest stable versions that install together: torch, transformers, lightning, spacy, nltk, python-Levenshtein, sentencepiece, safetensors, huggingface_hub, and easse via git URL. Start with easse at its current HEAD; Task 3 decides whether to fall back to `8df369f`.
  - Extras: `search = ["optuna"]`, `dev = ["pytest"]`.
  - `[tool.pytest.ini_options] markers = ["slow: needs model weights"]`.
  - `LICENSE`: the official CC BY-NC 4.0 legalcode text.
- [ ] **Step 4:** Port the helpers into `src/ts_t5/data.py` from `source/resources.py` and `source/helper.py`. `REPO_DIR = Path(__file__).resolve().parents[2]` (`# ponytail: repo-anchored paths; editable install only, add a TS_T5_HOME env var if wheels are wanted`). Leave out the GloVe download, `get_temp_filepath` and tuning dirs; nothing calls them. Run the tests. Expected: PASS.
- [ ] **Step 5:** `git commit -m "build: pyproject, licence, ts_t5.data"`

### Task 2: Control-token features

**Files:**
- Create: `src/ts_t5/features.py` (port of `source/preprocessor.py`), `tests/test_features.py`, `tests/fixtures/processed_pairs.tsv`

**Interfaces:**
- Consumes: `ts_t5.data` paths, `read_lines`, `generate_hash`.
- Produces: `Preprocessor(features_kwargs: dict)` with `.encode_sentence(s) -> str`, `.encode_sentence_pair(c, s) -> str`, `.hash`, `.preprocess_dataset(dataset) -> Path`; constants `FEATURES_4TOKENS`, `FEATURES_5TOKENS` (the F4 and F5 values above).

- [ ] **Step 1: Build the fixture.**
  - Find the `resources/processed_data/<hash>/wikilarge/` dir whose hash equals the old `Preprocessor(<0.8 kwargs>).hash`, for the 4-token feature set.
  - Write the first 30 lines of `wikilarge.train.complex` from `resources/datasets/wikilarge/`, the matching `.simple` lines, and the processed line as a 3-column TSV.
  - If neither processed dir matches, generate the fixture with the old code: `git show main:source/preprocessor.py`, run it in the new venv, and note that in the test docstring.
- [ ] **Step 2: Write the failing tests**

```python
def test_encode_sentence_exact_string():
    s = 'The cat sat on the mat .'
    assert Preprocessor(FEATURES_4TOKENS).encode_sentence(s) == 'C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s
    assert Preprocessor(FEATURES_5TOKENS).encode_sentence(s) == 'W_1.05 C_0.95 L_0.75 WR_0.75 DTD_0.75  ' + s

def test_encode_pair_matches_2021_processed_data():
    p = Preprocessor({k: {'target_ratio': 0.8} for k in FEATURES_4TOKENS})
    rows = [l.split('\t') for l in read_lines(FIXTURE)]
    mismatches = [c for c, s, want in rows if p.encode_sentence_pair(c, s) != want]
    assert not mismatches, mismatches[:3]
```

  The double space before the sentence is the original behaviour; keep it.
- [ ] **Step 3:** Port the code. Move `nltk.download('stopwords')` and the spaCy model download into lazy, cached getters. Drop `spacy.cli.link`, which spaCy 3 does not have. Keep `round` as `'%.2f' % val`, keep `get_word2rank` reading `DUMPS_DIR/glove.42B.300d.pk`, and keep the `globals()[class_name]` lookup. Drop dead code: `get_complexity_score*`, `get_word_frequency`.
- [ ] **Step 4:** Run `pytest tests/test_features.py`. Expected: PASS. If the pair test fails only on `DTD_` values, try the 2021-era `en_core_web_sm` version. If it still fails, record the mismatch rate and the cause in a `REPRODUCE.md` stub, mark the test `xfail` with that reason, and tell the author. Inference does not depend on it.
- [ ] **Step 5:** `git commit -m "feat: port control-token features"`

### Task 3: Scoring with easse

**Files:**
- Create: `src/ts_t5/evaluate.py` (scoring half), `tests/test_scoring.py`

**Interfaces:**
- Produces: `score_file(test_set: Literal['turkcorpus', 'asset'], sys_path: Path) -> dict[str, float]` (keys `sari`, `bleu`, `fkgl`), and `EXPECTED: dict[str, dict[str, dict[str, float]]]` with the README table, keyed `EXPECTED['4tokens']['turkcorpus']['sari']`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize('model,h', [('4tokens', 'd859b9c4aab9f7a9c719167728399278'),
                                     ('5tokens', '267a9c5912b4c0e37bf8bf7260f7576d')])
@pytest.mark.parametrize('test_set', ['turkcorpus', 'asset'])
def test_published_outputs_score_as_readme(model, h, test_set):
    got = score_file(test_set, REPO_DIR / f'system_outputs/TS_T5_{model}/{h}_{test_set}.test.txt')
    want = EXPECTED[model][test_set]
    assert abs(got['sari'] - want['sari']) <= 0.1
    assert abs(got['bleu'] - want['bleu']) <= 0.5 and abs(got['fkgl'] - want['fkgl']) <= 0.5
```

- [ ] **Step 2:** Run it. Expected: FAIL (no module).
- [ ] **Step 3:** Implement `score_file` with `easse.cli.evaluate_system_output(test_set=f'{test_set}_test', sys_sents_path=str(p), lowercase=True)`.
- [ ] **Step 4:** Run it. Expected: PASS. If scores drift past the margins, pin easse to `8df369f` in `pyproject.toml`, re-run, and write the before/after numbers into `docs/REPRODUCE.md`. This test separates easse drift from model drift before any model runs.
- [ ] **Step 5:** `git commit -m "feat: easse scoring pinned to README numbers"`

### Task 4: Simplify (inference)

**Files:**
- Create: `src/ts_t5/simplify.py`, `tests/test_simplify.py`
- Modify: `src/ts_t5/evaluate.py` (add `evaluate_model`)

**Interfaces:**
- Consumes: `Preprocessor`, `FEATURES_*`, `score_file`.
- Produces:
  - `Simplifier(model: str | Path, features_kwargs: dict | None = None, device: str | None = None)` with `.max_len: int`, `.simplify(sentence: str) -> str` (`""` when filtered), and `.simplify_file(src: Path, dst: Path) -> Path`.
  - `features_kwargs` defaults to `config.ts_t5['features_kwargs']` from the model's `config.json`, then to `FEATURES_4TOKENS`.
  - `max_len` reads `config.ts_t5['max_seq_length']`, then `params.json` in a local dir, else 80.
  - `evaluate_model(model, test_set, features_kwargs, out_dir) -> dict` writes `out_dir/f'{generate_hash(features_kwargs)}_{test_set}.test.txt'` and returns `score_file(...)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_simplify_file_keeps_line_count(tmp_path, monkeypatch):
    s = Simplifier.__new__(Simplifier)                     # no weights
    monkeypatch.setattr(s, 'simplify', lambda x: '' if 'same' in x else x.upper())
    src = tmp_path / 'in.txt'; write_lines(['a', 'same', "it ''s"], src)
    out = s.simplify_file(src, tmp_path / 'out.txt')
    assert read_lines(out) == ['A', '', 'IT "S']

@pytest.mark.slow
def test_simplify_real_model():
    s = Simplifier(os.environ.get('TS_T5_MODEL', 'kcsheang/ts-t5-base-4tokens'))
    src = ['The incumbent was defeated in the election by a substantial margin .', ...]  # 3 sentences
    out = [s.simplify(x) for x in src]
    assert all(out) and all(o != x for o, x in zip(out, src))
```

- [ ] **Step 2:** Run `pytest tests/test_simplify.py -m "not slow"`. Expected: FAIL.
- [ ] **Step 3:** Implement the generate body from `source/evaluate.py:generate`:
  - Keep the same tokenizer call (`padding='max_length', truncation=True`) and the same `generate` kwargs.
  - Keep the post-filter: drop the output when `sent.lower() == encoded.lower()`.
  - `simplify_file` writes one line per input and then applies the `"''"` → `'"'` replacement.
  - Leave in `torch.manual_seed(12)` at construction.
  - Device order: cuda, then mps, then cpu.
  - `# ponytail: one sentence per generate() call like 2021; batch only after the gate shows identical outputs`.
- [ ] **Step 4:** Run the tests. Expected: the fast test passes. Running the slow test needs `TS_T5_MODEL=<local 4-token dir>` (Task 8).
- [ ] **Step 5:** `git commit -m "feat: Simplifier and evaluate_model"`

### Task 5: Training on Lightning 2 and checkpoint conversion

**Files:**
- Create: `src/ts_t5/model.py`, `src/ts_t5/train.py`, `scripts/convert_ckpt_to_hf.py`, `configs/4tokens.json`, `configs/5tokens.json`, `configs/smoke.json`, `tests/test_train.py`

**Interfaces:**
- Produces:
  - `train(config: dict) -> Path` returns `output_dir`. It writes `params.json` and checkpoints there, and logs `train_loss` per step to `output_dir/metrics.csv` (Lightning `CSVLogger`).
  - `convert(ckpt: Path, out_dir: Path, params: dict) -> Path` keeps `model.`-prefixed keys, strips the prefix and saves `model.safetensors`, the t5-base tokenizer, and `config.ts_t5 = {'max_seq_length': int, 'features_kwargs': dict}`.

- [ ] **Step 1: Write configs.**
  - `4tokens.json` and `5tokens.json` copy the D8 `params.json` values, with numbers as numbers and `features_kwargs` as a JSON object in the original order. 4-token: batch 32, accum 4. 5-token: batch 64, accum 1. Both: max_seq_length 80, lr 3e-4, wd 0.1, eps 1e-8, warmup 5, epochs 5, seed 12, all 0.8 targets.
  - Take `dataset` from the D8 `logs.txt`. Default is `wikilarge`; `processed_data/` also has `wiki-auto`, so check.
  - `smoke.json`: 4-token features, `model_name: t5-small`, `max_steps: 100`, `train_sample_size` giving 800 pairs, batch 8, one validation pass on 16 valid sentences.
- [ ] **Step 2: Write the failing tests**

```python
@pytest.mark.slow
def test_smoke_train_converts_and_loads(tmp_path):
    out = train({**load_config('smoke'), 'output_dir': tmp_path})
    loss = [float(r['train_loss']) for r in csv.DictReader(open(out / 'metrics.csv')) if r['train_loss']]
    assert mean(loss[-10:]) < mean(loss[:10])
    hf = convert(next(out.glob('*.ckpt')), tmp_path / 'hf', json.load(open(out / 'params.json')))
    assert Simplifier(hf).simplify('This is a fairly long and complicated sentence .') is not None

def test_convert_rejects_bad_state_dict(tmp_path):
    torch.save({'state_dict': {'wrong.prefix': torch.zeros(1)}}, tmp_path / 'x.ckpt')
    with pytest.raises(ValueError, match='missing'):
        convert(tmp_path / 'x.ckpt', tmp_path / 'hf', {'model_name': 't5-small', 'max_seq_length': 80, 'features_kwargs': {}})
```

- [ ] **Step 3:** Port `T5FineTuner` and the datasets to `lightning`:
  - Optimizer: `torch.optim.AdamW` with the same two param groups.
  - LR scheduler: `get_linear_schedule_with_warmup`, `total = trainer.estimated_stepping_batches`, returned from `configure_optimizers` with `interval: 'step'`. This replaces the manual `optimizer_step` override and the `self.opt.zero_grad()` call.
  - Drop apex and `amp_level`. Use `accelerator='auto'`. Drop `custom_loss` (it was off in both paper runs).
  - Keep validation as `1 - SARI/100` on TurkCorpus valid, with the same names `val_loss` and `checkpoint-{epoch}`, and `save_top_k=5`.
  - Use `num_workers=0` on macOS.
  - `convert`: load with `torch.load(..., weights_only=False)`, because it is a trusted local file. Raise `ValueError` listing missing or unexpected keys; allow only `lm_head.weight`.
- [ ] **Step 4:** Run `pytest tests/test_train.py`. Expected: PASS. Record the wall time and device for `REPRODUCE.md`.
- [ ] **Step 5:** `git commit -m "feat: Lightning 2 training, ckpt to HF conversion"`

### Task 6: Token search

**Files:**
- Create: `src/ts_t5/search.py` (port of `find_best_tokens.py`; drop `hyperparameters_tuning.py`, whose training-time tuning is not part of the release)

**Interfaces:**
- Produces: `search(model: str | Path, n_trials: int, test_set: str = 'turkcorpus', phase: str = 'valid', tokens: tuple[str, ...] = ('C', 'L', 'WR', 'DTD')) -> dict`. It returns the best `features_kwargs` and the best SARI, using the search space and step from `source/find_best_tokens.py`.

- [ ] **Step 1:** Port it, with optuna imported inside the function so the extra stays optional. Valid-phase scoring uses `corpus_sari` against the 8 Turk refs (or the 10 ASSET refs), as the old `evaluate()` did.
- [ ] **Step 2:** Check by hand with the local 4-token model: `ts-t5 search --model <dir> --trials 2 --limit 20` finishes and prints a kwargs dict. Add `limit: int | None` to cap sentences. No test: it is a thin wrapper over pieces already tested.
- [ ] **Step 3:** `git commit -m "feat: optuna control-token search"`

### Task 7: CLI and removal of the old layout

**Files:**
- Create: `src/ts_t5/cli.py`, `tests/test_cli.py`
- Delete: `source/`, `scripts/*.py` wrappers, `requirements.txt`
- Modify: `TS_T5.ipynb` (install with `pip install -e .`, load from HF, call `Simplifier`)

**Interfaces:**
- Produces `main(argv: list[str] | None = None) -> int` with these subcommands:
  - `train --config NAME|PATH`
  - `evaluate --model M [--tokens W=1.05,C=0.95,...] [--test-set turkcorpus|asset|all] [--out DIR]`, which prints a SARI/BLEU/FKGL table
  - `simplify --model M [--tokens ...] (SENTENCE | --input F --output F)`
  - `search --model M --trials N [--limit K]`
- `parse_tokens(s: str) -> dict` maps `W C L WR DTD` to the feature class names in the order given.

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_tokens_keep_order():
    assert list(parse_tokens('W=1.05,C=0.95,L=0.75,WR=0.75,DTD=0.75')) == list(FEATURES_5TOKENS)
    assert parse_tokens('C=0.95,L=0.75,WR=0.75,DTD=0.75') == FEATURES_4TOKENS

def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit): main(['--help'])
    assert all(c in capsys.readouterr().out for c in ('train', 'evaluate', 'simplify', 'search'))
```

  `parse_tokens` must produce floats equal to the defaults, so `str(0.95) == '0.95'` and the hash matches.
- [ ] **Step 2:** Run them. Expected: FAIL. Implement with `argparse`. Run them again. Expected: PASS.
- [ ] **Step 3:** Delete the old files. Check: `grep -rn "from source" src tests scripts` finds nothing, and `pytest -m "not slow"` passes.
- [ ] **Step 4:** `git commit -m "feat: ts-t5 CLI; remove source/ and wrappers"`

### Task 8: Reproduction gate (local, no uploads)

**Files:**
- Create: `docs/REPRODUCE.md` (gate section), `experiments/` (untracked)

- [ ] **Step 1: Pick the weights.**
  - Download both Drive zips (ids in the deleted `source/download_data.py`: `19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE` 4-token, `19s68NxHybYta0t-smhWKrGZxUYyRlgHA` 5-token) into `experiments/drive/`.
  - `shasum -a 256` the `pytorch_model.bin` files against copies of the D8 files.
  - Copy D8 into `experiments/d8/` with `cp`, never `mv`.
  - Record all hashes in `REPRODUCE.md`.
- [ ] **Step 2:** Convert the 5-token checkpoint (`checkpoint-epoch=4-val_loss=0.83.ckpt`) with `scripts/convert_ckpt_to_hf.py`. Also convert `epoch=3` only if `epoch=4` fails the gate. Wrap the 4-token `pytorch_model.bin` with the same `config.ts_t5` block, as `model.safetensors`. Put the results in `experiments/hf/{4,5}tokens/`.
- [ ] **Step 3:** Run `ts-t5 evaluate --model experiments/hf/4tokens --test-set all` and the same for 5tokens. Expected: every number within the margins in Global Constraints.
- [ ] **Step 4:** Diff each regenerated output against `system_outputs/TS_T5_*tokens/<hash>_*.test.txt`, and record the share of identical lines.
- [ ] **Step 5:** Check the README numbers against the paper's tables. The author supplies the PDF, or it is fetched from the ACL Anthology. Any difference goes in the report, not silently resolved.
- [ ] **Step 6:** Run `TS_T5_MODEL=experiments/hf/4tokens pytest -m slow`. Expected: PASS.
- [ ] **Step 7:** **Gate decision.** If all pass, write the gate table into `REPRODUCE.md` and commit: `git commit -m "docs: reproduction gate results"`. If any fail, stop: report the numbers, the Drive/D8 hash comparison and the identical-line share to the author, and do not continue to Task 11.

### Task 9: Results and findings docs

**Files:**
- Create: `docs/RESULTS.md`, `docs/ABLATIONS.md`, `docs/FINDINGS.md`, `docs/EXPERIMENTS.md`, `docs/figures/*`, `scripts/make_docs_tables.py`
- Modify: `docs/REPRODUCE.md` (full)

- [ ] **Step 1:** `scripts/make_docs_tables.py` scores every file in `system_outputs/` with `score_file` and prints markdown rows. The baseline files hold TurkCorpus outputs; check line counts (359) before scoring. `RESULTS.md` holds that table plus 5 example simplifications from the gate outputs, each with its source file noted.
- [ ] **Step 2:** `ABLATIONS.md`: parse D8 `workspace/TS_T5/results/for_ablations.txt`.
  - Map each output-file hash back to its token settings with `generate_hash` over candidate `features_kwargs`: every subset and order of the 5 tokens over the value grid seen in the D8 score logs (`score_*_log.txt` print the values).
  - Rows with no match are dropped and counted in a footnote.
- [ ] **Step 3:** `EXPERIMENTS.md`: one row per readable D8 run (`workspace/TS_T5/experiments`, `results/exp_*`, `TS_backup/TS_T5.2/experiments/old`): id, model, dataset, tokens, epochs, best val_loss from `logs.txt`, and status. Builder picks which runs to list.
- [ ] **Step 4:** `FINDINGS.md`: per-token effect from the ablation rows, effect of the 5th token (gate numbers), search results, failure cases from the gate outputs, and negative results. Every number cites its file.
- [ ] **Step 5:** Regenerate the D8 `plots/` figures as PNG into `docs/figures/` from the scripts there (copied, not run in place).
- [ ] **Step 6:** `REPRODUCE.md` covers:
  - The install commands and the exact `ts-t5 evaluate` commands.
  - Original hardware and run times from D8 `logs.txt` / `params.json`.
  - The smoke-train time from Task 5.
  - The easse note from Task 3.
  - The sentence "The full training was not re-run on the new stack."
- [ ] **Step 7:** `git commit -m "docs: results, ablations, findings, experiments"`

### Task 10: Human evaluation, anonymised

**Files:**
- Create: `scripts/anonymise_human_eval.py`, `tests/test_anonymise.py`, `human_evaluation/` (CSVs, `HIT_designs/`), `docs/HUMAN_EVALUATION.md`

**Interfaces:**
- Produces: `anonymise(df: pandas.DataFrame, salt: str) -> pandas.DataFrame` and `check_no_ids(paths: list[Path]) -> None`, which raises `ValueError` on a pattern match. Add `pandas` to the `dev` extra only.

- [ ] **Step 1: Write the failing test**

```python
def test_anonymiser_leaves_no_worker_id(tmp_path):
    df = pd.DataFrame({'WorkerId': ['A1B2C3D4E5F6G7'], 'AssignmentId': ['3ABCDEF'], 'HITId': ['3XYZ'],
                       'Answer.comment': ['my email is x@y.z'], 'Answer.fluency': [4]})
    out = anonymise(df, salt='s')
    assert 'Answer.comment' not in out and out['WorkerId'][0] != 'A1B2C3D4E5F6G7'
    assert out['WorkerId'][0] == anonymise(df, salt='s')['WorkerId'][0]  # stable
    out.to_csv(tmp_path / 'o.csv'); check_no_ids([tmp_path / 'o.csv'])
    with pytest.raises(ValueError): check_no_ids([_write(tmp_path, 'A1B2C3D4E5F6G7')])
```

- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement it.
  - The ID columns become `sha256(salt + id)[:12]`.
  - Keep `Input.*` and the numeric `Answer.*` rating columns. Drop free-text `Answer.*` and every other column.
  - Salt from env `TS_T5_SALT`, required. Store it in `~/.config/ts_t5/salt`, never in the repo.
- [ ] **Step 4:** Run the test. Expected: PASS. Then run the script on the D8 `TS_T5 Human evaluation/results/` copies, followed by `check_no_ids(human_evaluation/**)`. Also grep the copied `HIT_designs/` and `.xlsx`-derived CSVs; review them by hand for names or emails.
- [ ] **Step 5:** `HUMAN_EVALUATION.md` covers:
  - The protocol, taken from D8 `TS_T5 Human Evaluation.md` and the HIT design.
  - Aggregate scores per system, computed from the anonymised CSVs.
  - The note that the raw exports under `Human evaluation/results/raw from mTurk/` predate this release.
- [ ] **Step 6:** `git commit -m "feat: anonymised human evaluation"`

### Task 11: Model cards and Hub upload script

**Files:**
- Create: `model_cards/ts-t5-base-4tokens.md`, `model_cards/ts-t5-base-5tokens.md`, `scripts/push_to_hub.py`, `tests/test_cards.py`

**Interfaces:**
- Produces: `push(model_dir: Path, repo_id: str, card: Path, private: bool = True, dry_run: bool = True) -> None`. A real upload needs `--no-dry-run`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize('name,feats', [('4tokens', FEATURES_4TOKENS), ('5tokens', FEATURES_5TOKENS)])
def test_card_input_example_matches_preprocessor(name, feats):
    card = (REPO_DIR / f'model_cards/ts-t5-base-{name}.md').read_text()
    s = 'The cat sat on the mat .'
    assert '"simplify: ' + Preprocessor(feats).encode_sentence(s) + '"' in card
    assert 'license: cc-by-nc-4.0' in card and 'base_model: t5-base' in card
    for v in EXPECTED[name].values(): assert f"{v['sari']:.2f}" in card
```

- [ ] **Step 2:** Write the cards with every section listed in the spec's "Hugging Face model cards" section. The eval table and hyperparameters come from `REPRODUCE.md` and the configs, and the BibTeX comes from the README. Run the test. Expected: PASS.
- [ ] **Step 3:** Implement `push` with `huggingface_hub.HfApi.create_repo(private=private, exist_ok=True)` + `upload_folder`, uploading `model.safetensors`, `config.json`, the tokenizer files and the card as `README.md`. Dry run prints the file list. Check: `python scripts/push_to_hub.py ... ` (dry run) lists exactly those files.
- [ ] **Step 4:** `git commit -m "feat: model cards and push script"`

### Task 12: README, citation, CI

**Files:**
- Modify: `README.md`, `.gitignore`
- Create: `CITATION.cff`, `.github/workflows/tests.yml`

- [ ] **Step 1:** Rewrite `README.md`:
  - Opening: what it is, then a 3-line HF quickstart with `transformers` and the literal input example.
  - `pip install -e .` and the `ts-t5` commands.
  - The results table with a link to `docs/`.
  - A note that the Google Drive links are retired.
  - BibTeX and licence.
- [ ] **Step 2:** Write `CITATION.cff` from the README BibTeX. Validate it: `uvx cffconvert --validate`.
- [ ] **Step 3:** Workflow: ubuntu, Python 3.11, `pip install -e '.[dev]'`, `pytest -m "not slow"`. Locally, a fresh `uv venv` in `/tmp` clone with `pip install -e '.[dev]' && pytest -m "not slow"` must pass. That is the "clean machine" anti-goal check.
- [ ] **Step 4:** `git commit -m "docs: README, CITATION.cff, CI"`

### Task 13: Release (each step asks the author first)

- [ ] **Step 1:** Ask, then `git push -u origin release/hf-models`. Check that CI is green.
- [ ] **Step 2:** Ask, then `push_to_hub.py --no-dry-run` for both models with `private=True`. The author reviews the cards on HF. After that, run the slow test against `kcsheang/ts-t5-base-4tokens` loaded from the Hub.
- [ ] **Step 3:** Ask, then make both repos public (`HfApi.update_repo_settings(private=False)`), and create the Collection with both models and the paper.
- [ ] **Step 4:** Ask, then merge `release/hf-models` into `main` and push.
