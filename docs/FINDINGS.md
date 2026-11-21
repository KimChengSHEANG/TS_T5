# Findings

What we learned while preparing the release. Every number names its source.
Short names for sources: "ablation file" is `/Volumes/D8/workspace/TS_T5/results/for_ablations.txt` (tables in
`docs/ABLATIONS.md`); "D8" is the read-only 2020-22 backup under `/Volumes/D8/workspace`.

## Findings about the legacy code

### 1. Legacy validation paired mismatched files, so the logged `val_loss` is not a SARI score

- In the legacy `ValDataset` (`source/model.py` in the first commit `cc21371`, lines 319-327), the source is
  `<dataset>.valid.complex` and the references are always the eight TurkCorpus validation files.
- For `dataset = wikilarge` that pairs `wikilarge.valid.complex` (992 lines) with `turkcorpus.valid.simple.turk.0-7`
  (2000 lines each). The pairs go by line index. They are not the same sentences.
  Line counts: `resources/datasets/wikilarge/wikilarge.valid.complex` 992, `.../turkcorpus/turkcorpus.valid.complex` 2000.
- Evidence in the logs. The validation loop prints one SARI per batch (`Sari score:`). Median per-batch SARI:
  - March 2021 run `exp_1614936426182236_C_WR_L_DTD` (`logs.txt`, 560 batches, 112 per round): 43.3 (range 36.6-48.6).
  - May 2021 run `exp_1622148499982082_4tokens` (`logs.txt`, 155 batches, 31 per round, 31 x 32 = 992): 17.7 (range 12.3-22.0).
  - The May run validates on 992 inputs, which is the WikiLarge validation size, so the pairing is the mismatched one.
- Effect. `val_loss = 1 - SARI/100` is about 0.82 (checkpoint names such as `checkpoint-epoch=4-val_loss=0.83.ckpt`)
  and is noise around a meaningless number. Checkpoint selection used it. The May 2021 hyperparameter study
  (`docs/EXPERIMENTS.md`) reports `val_loss` of 0.82-0.88, the same scale. Compare `val_loss` only inside one era (`docs/EXPERIMENTS.md`, "Reading the columns").
- New code. `ts-t5 train` validates on `turkcorpus.valid` (aligned: 2000 inputs, 2000 refs per file) and logs
  `val_loss = 1 - SARI/100`, averaged over batches (`src/ts_t5/model.py`, `validation_step`). This keeps the legacy
  definition of the number (batch-averaged SARI, not corpus SARI) and fixes the pairing.
  A retrain with the new code can therefore pick different checkpoints than the 2021 runs did.
  The full training was not re-run on the new stack, so we have no measurement of that effect.

### 2. The legacy token search was not a valid search, and the default values cannot be traced to it

- Legacy `find_best_tokens` (Optuna, maximise SARI, each token in 0.20-1.50 with step 0.05). Two versions exist.
  - In the repo's last pre-refactor code (commit `5989c49^`), `find_best_tokens.py` calls
    `evaluate_on_TurkCorpus(features_kwargs, 'valid', 'exp_1621628499406503')`. That function generates on
    `turkcorpus.valid.complex` (2000 lines) but scores with `evaluate_system_output(test_set="turkcorpus_test", ...)`
    (`source/evaluate.py`, line 226 of that version), i.e. against 359 test references. The objective mixes splits.
  - The copy in D8 (`TS_T5/source/find_best_tokens.py`) differs: it calls
    `evaluate_on(TURKCORPUS_DATASET, features_kwargs, 'test', 'exp_1621628499406503')`. If this is the code that made the
    saved study, the search ran on the **test** set. We could not date the file, so this is not confirmed.
- The saved study is `TS_backup/TS_T5.2/experiments/tuning_logs/old/Tokens_study.db` (read as a copy;
  `python scripts/make_docs_tables.py search`). 239 complete trials, 2021-05-24 to 2021-05-25.
  The SARI values (24.8 to 43.61) are the size of TurkCorpus test scores, which fits the D8 code.

Top 10 trials of the study:

| Rank | Trial | C | L | WR | DTD | SARI |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 83 | 0.90 | 0.80 | 0.55 | 0.80 | 43.61 |
| 2 | 235 | 0.95 | 0.75 | 0.50 | 0.75 | 43.51 |
| 3 | 264 | 0.95 | 0.75 | 0.55 | 0.65 | 43.50 |
| 4 | 207 | 0.95 | 0.75 | 0.50 | 0.70 | 43.49 |
| 5 | 250 | 0.95 | 0.75 | 0.50 | 0.65 | 43.48 |
| 6 | 79 | 0.95 | 0.80 | 0.50 | 1.25 | 43.44 |
| 7 | 210 | 0.95 | 0.70 | 0.50 | 0.70 | 43.40 |
| 8 | 30 | 0.90 | 0.85 | 0.55 | 0.90 | 43.39 |
| 9 | 163 | 0.95 | 0.75 | 0.65 | 1.35 | 43.39 |
| 10 | 39 | 0.90 | 0.80 | 0.75 | 0.60 | 43.37 |

What the study shows:

- No trial used the released defaults (`C 0.95, L 0.75, WR 0.75, DTD 0.75`). The default point was never sampled.
- With C 0.95 and L 0.75 held, WR 0.50-0.55 gave 43.20-43.51. WR 0.70-0.85 gave 42.84-43.25 (12 trials with C 0.95
  and L 0.75, same database). At WR 0.50, DTD 0.65, 0.70 and 0.75 gave 43.48, 43.49 and 43.51, so DTD changed little there.
  The study prefers a lower WR than the default.
- The best trial is 0.16 above a D8 run whose ID is 27 minutes older than the study's model
  (`exp_1621626872354637`, 43.45 at the defaults, `TS_T5/outputs/exp_1621626872354637/score_d859...turkcorpus_test_log.txt`).
  We do not know if the two are the same training run. A gap of 0.16 is inside the run-to-run spread we cannot
  measure (see "What we could not test").
- The defaults are older than the study. The 4-token model of December 2020 (`exp_1607800772213084_C_WR_L_DTD`)
  already has output files at the default values, and its ablation rows compare 0.75 with 0.85 for L, WR and DTD
  (`docs/ABLATIONS.md`, section `exp_1607800772213084_C_WR_L_DTD`).

How the defaults were chosen: **cannot be confirmed.** We found no log of a search before the study that would
select `C 0.95, L 0.75, WR 0.75, DTD 0.75`. The ablation file shows a coarse manual grid (0.5, 0.75, 0.85, 1.0 for
L, WR, DTD together with C fixed at 0.95; and single-token sweeps from 0.25 to 1.15 in `docs/ABLATIONS.md`).
It is consistent with a hand-picked setting, but nothing in D8 says so. The paper text (Table 6) refers to
searched tokens per test set; we did not find the matching logs.

The new `ts-t5 search` (`src/ts_t5/search.py`) scores validation outputs against validation references, which is the
evident intent. Its values are not comparable with the 2021 study.

### 3. Dependency-tree depth (DTD) drifts between spaCy 2 and spaCy 3

- The 2021 processed training data was made with spaCy 2.x `en_core_web_sm`. spaCy 3 models parse differently.
- On the 30-row fixture in `tests/test_features.py`, `C_`, `L_` and `WR_` tokens match exactly and `DTD_` differs on
  11/30 rows with spaCy 3.8 (`en_core_web_sm` 3.8.0). Other versions: 3.0.0 10/30, 3.1.0 8/30, 3.7.1 11/30
  (`docs/REPRODUCE.md`, first section). The test is marked `xfail`.
- Training with the new stack would therefore see slightly different DTD tokens than the 2020-21 runs.
- Inference is not affected. It encodes the source sentence with fixed target ratios and never computes DTD on the
  source (`Preprocessor.encode_sentence`). The reproduction gate numbers confirm this (`docs/REPRODUCE.md`).

### 4. The 5-token Drive `params.json` has no `features_kwargs`

- `experiments/drive/5tokens/model_5tokens/params.json` (and the same file in D8,
  `TS_backup/TS_T5.2/experiments/exp_1601582154282931_best/params.json`) lists no `features_kwargs`, `dataset`,
  `custom_loss`, `train_sample_size` or `valid_sample_size`. The 4-token Drive file does have `features_kwargs`.
- The token set of the released 5-token model is therefore inferred from scores, not read from the params:
  - The output hash `267a9c5912b4c0e37bf8bf7260f7576d` equals `generate_hash` of `{W: 1.05, C: 0.95, L: 0.75, WR: 0.75, DTD: 0.75}`
    (`ts_t5.features.FEATURES_5TOKENS`).
  - Fresh generation with those tokens reproduces the README: TurkCorpus 43.33 / 65.53 / 5.55 vs 43.34 / 65.54 / 5.55,
    ASSET 45.27 / 72.20 / 5.91 vs 45.26 / 72.19 / 5.92 (`docs/REPRODUCE.md`, gate table).
  - The original score logs print the same five tokens
    (`exp_1601582154282931_best/outputs.6/score_267a...turkcorpus_test_log.txt`).
- The training target value (0.8 for every token, as in the other runs) is not recorded for this run. The release
  config uses 0.8 on the assumption that it matches. We could not check that.

## What each token does

Single-token runs, from `docs/ABLATIONS.md` (ablation file, test sets, one run per point):

- No tokens (t5-base): TurkCorpus 37.56, ASSET 34.15 SARI (`exp_1601753330122698_base_no_tokens`).
- Best single token on TurkCorpus: L at 0.85 (40.91), then WR at 0.85 (40.45). C, W and DTD reach 38.9-39.2.
  The gains over no tokens are +3.4 (L), +2.9 (WR), +1.3 to +1.6 (C, W, DTD).
- ASSET prefers more compression than TurkCorpus. The best values are C 0.5, DTD 0.5, WR 0.25.
- Adding C to the WR+L model is the largest step among the multi-token models (+2.5 SARI TurkCorpus).
  The 4th and 5th tokens add little (`docs/ABLATIONS.md`, "Adding tokens at a shared value").
- Token order is part of the input. Swapping L and WR in the 4-token model costs 0.70 SARI on TurkCorpus and
  3.01 on ASSET (43.31 to 42.61; 44.91 to 41.90).
- Dropping tokens from the 5-token model at inference costs 1.08 SARI or more on TurkCorpus.

### What the length tokens do

Figures in `docs/figures/`. Generated with `python scripts/make_figures.py --data /Volumes/D8/workspace/TS_T5/plots/data`.
The inputs are 2021 system outputs on the TurkCorpus validation split from D8 `plots/data`. They come from 2021 models
whose weights we did not identify. Do not read them as the released weights.

- `char_ratio_4tok_vs_5tok.png`: output length against input length (char ratio), for C = 0.25, 0.5, 0.75, 1.0, other
  tokens 0.75 (and W 1.0 for the 5-token model). Median char ratio, 4-token / 5-token:
  0.37 / 0.46 at C 0.25, 0.59 / 0.74 at 0.5, 0.76 / 0.85 at 0.75, 1.02 / 1.01 at 1.0. The reference median is 0.98.
  So C steers length in the right direction. Outputs are longer than requested at low C (0.37 for 0.25, 0.59 for 0.5)
  and match at 0.75 and 1.0. The 5-token model with W 1.0 stays longer at low C, probably because W 1.0 asks for
  the same word count (we did not test this).
- `levenshtein_4tok_vs_5tok.png`: the same panels for Levenshtein similarity.
- `char_ratio_by_W.png`: a 5-token model given only C = 1.0 and W = x. Median char ratio 0.95, 1.40, 1.42, 1.64
  for W = 0.25, 0.5, 0.75, 1.0. Outputs are longer than the input for W of 0.5 or more. The model saw only two of
  its five tokens, which is outside normal use. We keep the figure because it comes from the D8 scripts. Do not rely on it.
- `dataset_char_ratio.png`: WikiLarge training pairs have much more spread in compression (many pairs below 0.5)
  than TurkCorpus references (a sharp peak at 1.0). The training pairs and the test references do not have the same
  length distribution.
- The score logs next to those outputs (`plots/data/{4,5}tokens/score_*_turkcorpus_valid_log.txt`, 2021 scoring script)
  give validation SARI. 4-token: C 0.25 29.94, 0.5 35.51, 0.75 41.32, 1.0 43.20. 5-token: 34.73, 40.45, 42.65, 43.00.
  The 5-token model loses less when C is low.

## The 5th token (W)

README numbers (`docs/RESULTS.md`), 5-token minus 4-token:

| Test set | SARI | BLEU | FKGL |
|---|---|---|---|
| TurkCorpus | -0.16 | -1.72 | -0.62 |
| ASSET | +0.21 | -0.05 | -0.41 |

- Paper numbers give the same pattern: TurkCorpus -0.31 SARI, ASSET +0.13 (Tables 2 and 5 in `docs/RESULTS.md`).
- The W token does not raise SARI on TurkCorpus. On ASSET the change is +0.2. The clearer effect is on readability:
  FKGL drops 0.4 to 0.6. BLEU falls 1.7 on TurkCorpus.
- With one run per model, differences of 0.2 SARI are not separable from noise (see "What we could not test").

## Failure cases

Hand-checked lines from the gate outputs `experiments/out_drive256/{4,5}tokens/*.test.txt` (untracked; the same lines
are in `system_outputs/`). TurkCorpus is `resources/datasets/turkcorpus/turkcorpus.test.complex`, ASSET is
`resources/datasets/asset/asset.test.orig`. Line numbers start at 1. We did not count error rates; these are examples.

**A fact the source does not support (added content).**
- TurkCorpus line 2. Source: "... Islam 's holiest city , which able-bodied Muslims are required to visit at least once in their lifetime ."
  4-token: "... All Muslims must visit Jeddah at least once in their lives **to become a Muslim**." The last part is false.
  The 5-token output is faithful here.
- ASSET line 40. Source: "... works towards popularizing interaction design as a mainstream discipline."
  4-token: "He also wants to make interaction design more popular **in the United States**."
- ASSET line 5 (5-token). It adds "**and speed**" to "pulling on the cord with all his strength".

**The meaning is changed.**
- TurkCorpus line 24. Source: "Both names became defunct in 2007 when they were merged into The National Museum of Scotland."
  4-token: "Both names **were not used again until 2007** when they became The National Museum of Scotland." This says the
  opposite about 2007. The 5-token output invents an acronym: "... ( **MNS** )".
- TurkCorpus line 3. "represent a hole in the methane cloud deck" becomes "be in the methane cloud deck" (4-token).
  The 5-token output drops "methane". Both add "the planet".

**Information is dropped or world knowledge is injected.**
- TurkCorpus line 229. Source: "a professional ice hockey team based in Montreal , Quebec , Canada ."
  Both models: "an ice hockey team in the **National Hockey League ( NHL )**". "professional", "Quebec" and "Canada"
  are gone. The league is not in the source.

**Unknown characters disappear.**
- TurkCorpus and ASSET line 14. "BZÖ" becomes "BZ" in both models and both test sets. Cause (checked): the T5
  tokenizer in `experiments/hf/4tokens` has no token for "Ö". `tokenizer("BZÖ")` gives `['▁B', 'Z', '<unk>', '</s>']`
  and `decode(..., skip_special_tokens=True)` removes the `<unk>`. In each test set, 2 of 359 sources contain such a
  character after tokenising (lines 14 and 145).

**Sentence splits start with a pronoun.** Many outputs split one sentence in two and open the second with "They" or "These"
(TurkCorpus lines 14 and 336, ASSET line 35). That is the intended behaviour of the length tokens. It can make the
second sentence depend on the first.

Overall: almost no output is a copy of its source. Ignoring spaces, case and punctuation, 1 (4-token) and 2 (5-token)
of 359 TurkCorpus outputs equal their source, and 0 (4-token) and 1 (5-token) of 359 ASSET outputs do.
Median output length is 0.93-0.95 of the input length in characters.
(Computed from the four gate output files.)

## Negative results

- The D8 retrains do not reproduce the README (`docs/REPRODUCE.md`): 18-22% identical lines and BLEU 1-2 points high.
  The released weights are the 2020 Drive runs, and the 2021 D8 runs are a documented later run, not the released model.
- On 19-21 May 2021, 18 D8 output folders (`TS_T5/outputs/exp_16214...` to `exp_16216...`) hold TurkCorpus scores at the
  default tokens. 15 of the 18 scored below 40 SARI (down to 16.64 with BLEU 0.00), 8 had BLEU above 90, which means
  near-copy output. Only 3 reached 42.77-43.45. We found no training logs or params for these runs, so we cannot say
  what went wrong in them.
- mT5-base (`exp_1616445446749616_mT5`, 4 tokens): TurkCorpus 43.14 / 70.34 / 6.50, ASSET 43.53 / 74.30 / 6.45
  (`docs/EXPERIMENTS.md`). No gain over t5-base on TurkCorpus (43.16 for the released 4-token model in the same era) and
  1.3 below on ASSET.
- 8 epochs instead of 5 (`exp_1615637544716827` vs `exp_1614936426182236`, same batch and length): TurkCorpus
  42.25 vs 42.79, ASSET 45.64 vs 45.45. No clear gain.
- Smaller model. t5-small with 5 tokens (`exp_1601752818273947_small`, ablation file): 40.83 / 85.12 / 6.78
  (TurkCorpus) and 39.13 / 86.08 / 7.00 (ASSET), against 43.00 and 45.04 for t5-base. t5-small with no tokens:
  34.50 and 29.85 against 37.56 and 34.15.
- The 5th token does not help on TurkCorpus (see above).
- The hyperparameter study (`TS_T5_study.db`) ranked a learning rate of 2.8e-6 best on a broken validation metric.
  Not used.
- The token search found values 0.1 to 0.2 SARI better on the test set than the released defaults, but it was run
  on the test set and the gain is smaller than we can measure. We did not change the defaults.

## What we could not test

- **Run-to-run noise.** Every row has one training run with seed 12 (`params.json`) and one decode. We did not
  measure the spread between seeds or between decodes. Differences of a few tenths of SARI are not claims.
- **Retraining on the new stack.** The full training was not re-run on the new stack. The smoke train (about 32 s on
  MPS, `configs/smoke.json`) only checks that the code runs.
- **How the default tokens were chosen** (finding 2).
- **Why the paper numbers are lower than the README** (`docs/RESULTS.md`), and why the original 2021 score logs of
  the released models (43.16 and 43.00 SARI on TurkCorpus) are below the re-scored README numbers (43.50 and 43.34).
- **Which model produced the D8 plot data.** The figure data are validation outputs of 2021 models we cannot name.
- **Human evaluation.** This document covers automatic scores only. It does not report human ratings.
