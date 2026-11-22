# Experiments

Training runs found in the 2020-22 D8 backups, one row per readable run directory. Built by
`python scripts/make_docs_tables.py experiments --d8 /Volumes/D8/workspace` (read-only).

Folders read:

- `/Volumes/D8/workspace/TS_T5/experiments` (4 runs; its `old/` folder is empty).
- `/Volumes/D8/workspace/TS_backup/TS_T5.2/experiments` (`all/`, `best_models/`, `keep/`, `old/` and the top level).
- There are no `results/exp_*` folders in the backup. `results/` holds only `for_ablations.txt`.

124 distinct runs were read (2 of them exist in both backups). Each run has `params.json`, `logs.txt`, or both.
Model, tokens, epochs, `max_seq_length` and batch come from `params.json`.
"Best val_loss" and "Status" come from `logs.txt`.

## Reading the columns

- **Tokens.** From `features_kwargs` in `params.json`. All training runs set every token to target 0.8 (`features_kwargs`
  is only used to generate during validation and to name the cache. Training inputs always carry the measured ratios). "not recorded" means `params.json` has no `features_kwargs`.
  For `exp_1601582154282931_best` (the 5-token Drive model) the token set is inferred from reproducing scores
  (`docs/FINDINGS.md`, finding 4).
- **Batch x accum.** `train_batch_size` times `gradient_accumulation_steps`.
- **Best val_loss.** The lowest epoch value of `val_loss`. If the progress bar in `logs.txt` shows `val_loss=`, that is used.
  Older logs only print one `Val_loss` per validation batch, so we averaged those per validation round and took the lowest.
  A dash means neither is in the log.
- **val_loss is not comparable across eras.** Runs from March 2021 (about 0.42-0.43) and from May 2021 on (0.82-0.92)
  were validated on different pairs of files. In the May runs the validation inputs and references were misaligned,
  so the number is not a SARI score. See `docs/FINDINGS.md`, finding 1.
  Within one era, lower is still better.
- **Status.** "finished" means `logs.txt` contains the line `training finished`. "no logs.txt (params only)" means the run
  folder keeps weights and `params.json`, but the training log was not kept.
- **Dataset.** Only some runs (June 2021 and later) record `dataset` in `params.json`; where present it is
  `wikilarge`. A dash means not recorded. The release configs assume WikiLarge. Two March 2021 runs have names that
  say `wikilarge_wikiauto` (WikiLarge plus Wiki-Auto); that comes from the folder name, not from `params.json`.

## Runs

| Run id | Date | Model | Dataset | Tokens | Epochs | max_seq | Batch x accum | Best val_loss (bar) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp_1601582154282931_best` | 2020-10-01 | t5-base | - | not recorded | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1601752818273947_small` | 2020-10-03 | t5-small | - | not recorded | 5 | 256 | 12x16 | - | no logs.txt (params only) |
| `exp_1601753330122698_base_no_tokens` | 2020-10-03 | t5-base | - | not recorded | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1602001102227427_small_no_tokens` | 2020-10-06 | t5-small | - | not recorded | 5 | 256 | 12x16 | - | no logs.txt (params only) |
| `exp_1607800772213084_C_WR_L_DTD` | 2020-12-12 | t5-base | - | C+L+WR+DTD | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611490086932860_DTD` | 2021-01-24 | t5-base | - | DTD | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611533249297539_WR_L` | 2021-01-25 | t5-base | - | WR+L | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611575591742851_C_WR_L` | 2021-01-25 | t5-base | - | C+WR+L | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611620290763589_W_C_WR_L` | 2021-01-26 | t5-base | - | W+C+WR+L | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611667703272792_C` | 2021-01-26 | t5-base | - | C | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611714803808738_W` | 2021-01-27 | t5-base | - | W | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611761596771104_L` | 2021-01-27 | t5-base | - | L | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1611808804849777_WR` | 2021-01-28 | t5-base | - | WR | 5 | 256 | 6x16 | - | no logs.txt (params only) |
| `exp_1614936426182236_C_WR_L_DTD` | 2021-03-05 | t5-base | - | C+L+WR+DTD | 5 | 256 | 18x16 | 0.427 | finished |
| `exp_1615235320272410_all_tokens_wikilarge_wikiauto` | 2021-03-08 | t5-base | - | W+C+L+WR+DTD | 5 | 256 | 18x16 | 0.422 | finished |
| `exp_1615637544716827_8epocs_wikilarge_wikiauto` | 2021-03-13 | t5-base | - | C+L+WR+DTD | 8 | 256 | 18x16 | 0.432 | finished |
| `exp_1616445446749616_mT5` | 2021-03-22 | google/mt5-base | - | C+L+WR+DTD | 5 | 256 | 4x16 | 0.423 | finished |
| `exp_1622063843255469_4tokens_b32` | 2021-05-26 | t5-base | - | C+L+WR+DTD | 5 | 80 | 32x4 | 0.818 | finished |
| `exp_1622148499982082_4tokens` | 2021-05-27 | t5-base | - | C+L+WR+DTD | 5 | 80 | 32x4 | 0.815 | finished |
| `exp_1622475676667979_small` | 2021-05-31 | t5-small | - | C+L+WR+DTD | 5 | 80 | 128x2 | 0.873 | finished |
| `exp_1622487333976136` | 2021-05-31 | t5-small | - | C+L+WR+DTD | 5 | 80 | 128x2 | 0.873 | finished |
| `exp_1622533103979907` | 2021-06-01 | t5-small | - | C+L+WR+DTD | 5 | 80 | 128x2 | 0.873 | finished |
| `exp_1622739349834393` | 2021-06-03 | t5-base | wikilarge | C+L+WR+DTD | 1 | 80 | 64x1 | 0.878 | finished |
| `exp_1622751002593370` | 2021-06-03 | t5-small | - | C+L+WR+DTD | 5 | 80 | 64x2 | 0.862 | finished |
| `exp_1622797913710174` | 2021-06-04 | t5-small | - | C+L+WR+DTD | 5 | 80 | 64x2 | - | finished |
| `exp_1622798169087856` | 2021-06-04 | t5-small | - | C+L+WR+DTD | 5 | 80 | 64x2 | - | finished |
| `exp_1622798261311710` | 2021-06-04 | t5-small | - | C+L+WR+DTD | 5 | 80 | 64x2 | 0.909 | finished |
| `exp_1622814894174641` | 2021-06-04 | t5-base | - | C+L+WR+DTD | 1 | 80 | 64x1 | 0.873 | finished |
| `exp_1622818645875244` | 2021-06-04 | t5-base | - | C+L+WR+DTD | 5 | 80 | 64x1 | 0.823 | finished |
| `exp_1622853103466272` | 2021-06-05 | t5-base | wikilarge | C+L+WR+DTD | 5 | 80 | 64x1 | 0.823 | finished |
| `exp_1622911289698489` | 2021-06-05 | t5-base | wikilarge | C+L+WR+DTD | 1 | 80 | 64x1 | 0.830 | finished |
| `exp_1623415175800762_5tokens` | 2021-06-11 | t5-base | - | W+C+L+WR+DTD | 5 | 80 | 64x1 | 0.827 | finished |
| `exp_1654762192055501` | 2022-06-09 | t5-base | - | W+C+L+WR+DTD | 5 | 80 | 16x1 | - | finished |
| `exp_1655114935938119` | 2022-06-13 | t5-base | - | W+C+L+WR+DTD | 5 | 80 | 16x1 | - | finished |
| 90 runs `exp_16226...`-`exp_16232...` | 2021-06-02 to 2021-06-10 | t5-small | - | C+L+WR+DTD | 1-3 | 80 | 64x1 | 0.859 to 0.921 | 83 finished, 7 not |


The last row groups 90 short t5-small runs (1 to 3 epochs, batch 64, `max_seq_length` 80) from 2021-06-02 to 2021-06-10.
They are quick tests of training code and settings. Their folder names are `exp_16226...` to `exp_16232...`.
We list them as one row because none has a documented purpose. Their best val_loss ranges from 0.859 to 0.921, and
83 finished (7 stopped before the final log line).

## Notes on specific runs

- `exp_1607800772213084_C_WR_L_DTD` (2020-12-12). The released 4-token model. `pytorch_model.bin` in
  `TS_backup/TS_T5.2/experiments/all/exp_1607800772213084_C_WR_L_DTD/` has sha256
  `e693d5bdaefdd614b441bb5c25b1d88725d71d698d5727c960c0d8ea7e8b5f65`, the same as the Drive zip.
  `params.json`: t5-base, `max_seq_length` 256, batch 6 x 16, 5 epochs, lr 3e-4. No `logs.txt`.
- `exp_1601582154282931_best` (2020-10-01). The released 5-token model, sha256
  `36091a14fbb5dfd0b0ecea20304df319f7ff92ae1044f120322461bf929d844f`, the same as the Drive zip. Its `params.json`
  has no `features_kwargs`. No `logs.txt`.
- Jan 2021 runs `exp_16114...` to `exp_16118...`. Single-token and small-subset ablation models
  (`docs/ABLATIONS.md`). Weights and `params.json` only.
- `exp_1614936426182236_C_WR_L_DTD` (2021-03-05). A 4-token retrain on a local machine (`output_dir` under
  `/Users/kimchengsheang/`), batch 18 x 16. Its test scores at the default tokens are in
  `TS_T5/outputs/exp_1614936426182236_C_WR_L_DTD/score_d859...` : TurkCorpus 42.79 / 59.69 / 5.07,
  ASSET 45.45 / 66.94 / 5.38. This is not the released model.
- `exp_1622148499982082_4tokens` (2021-05-27). The D8 4-token model that the reproduction gate rejected
  (65/359 identical lines against `system_outputs/`, BLEU 1.18 too high). `max_seq_length` 80, batch 32 x 4. The `output_dir` is a Google Colab path
  (`/content/drive/MyDrive/...`). The progress bar of its last epoch shows 2 h 28 min.
- `exp_1623415175800762_5tokens` (2021-06-11). The D8 5-token run. Only Lightning `.ckpt` files were kept
  (`checkpoint-epoch=3-val_loss=0.83.ckpt` and `checkpoint-epoch=4-val_loss=0.83.ckpt`). Also rejected by the gate.
- `exp_1615235320272410_all_tokens_wikilarge_wikiauto`, `exp_1615637544716827_8epocs_wikilarge_wikiauto`,
  `exp_1616445446749616_mT5` (March 2021). A 5-token run, an 8-epoch run, and an mT5 run (`google/mt5-base`,
  batch 4 x 16). Their default-token test scores are in the next section.
- `exp_1654762192055501`, `exp_1655114935938119` (June 2022). 5-token t5-base runs with batch 16, from
  `scripts/train_and_evaluate.minipc.py` (wandb run metadata: one GeForce RTX 3090). `logs.txt` has no validation numbers.
  We found no score files for them.

## Test scores at the default tokens (original 2021 pipeline)

Scores printed by the original scoring script (`score_<hash>_<set>_test_log.txt` in each run's `outputs*` folder),
at the default tokens: 4-token hash `d859b9c4...`, 5-token hash `267a9c59...`. SARI / BLEU / FKGL. These were not
re-scored here. Where a run has several `outputs*` folders we list the one named in the last column. Some folders hold
different generation runs with slightly different scores (for example `exp_1622148499982082_4tokens/outputs.1` gives
ASSET 45.10 / 73.13 / 6.18, `outputs` gives 44.75 / 72.71 / 6.18).

| Run | Tokens | TurkCorpus | ASSET | Folder |
|---|---|---|---|---|
| `exp_1601582154282931_best` (released 5-token) | 5 | 43.00 / 64.42 / 5.63 | 45.04 / 71.21 / 5.88 | `outputs.6` |
| `exp_1607800772213084_C_WR_L_DTD` (released 4-token) | 4 | 43.16 / 65.87 / 6.17 | 44.84 / 71.24 / 6.31 | `outputs.4` |
| `exp_1614936426182236_C_WR_L_DTD` | 4 | 42.79 / 59.69 / 5.07 | 45.45 / 66.94 / 5.38 | `TS_T5/outputs/exp_1614936426182236_C_WR_L_DTD` |
| `exp_1615235320272410_all_tokens_wikilarge_wikiauto` | 5 | 43.04 / 59.83 / 4.72 | 45.24 / 68.79 / 5.56 | `outputs` |
| `exp_1615637544716827_8epocs_wikilarge_wikiauto` | 4 | 42.25 / 58.38 / 5.16 | 45.64 / 64.89 / 5.48 | `outputs` |
| `exp_1616445446749616_mT5` | 4 | 43.14 / 70.34 / 6.50 | 43.53 / 74.30 / 6.45 | `outputs` |
| `exp_1622063843255469_4tokens_b32` | 4 | 43.37 / 68.72 / 6.23 | 44.99 / 73.47 / 6.30 | `outputs` |
| `exp_1622148499982082_4tokens` | 4 | 43.16 / 66.62 / 6.17 | 44.75 / 72.71 / 6.18 | `outputs` |
| `exp_1623415175800762_5tokens` | 5 | 43.19 / 64.57 / 5.60 | 45.07 / 71.76 / 5.82 | `outputs.1` |

Two things to note:

- The 5-token paper numbers (43.00 / 64.42 / 5.63 and 45.04 / 71.21 / 5.88) are the `outputs.6` row. They use the
  0.75 default, not 0.85. The paper's 4-token numbers (43.31 / 66.23 / 6.17, 44.91 / 71.96 / 6.32) come from that run's
  `outputs.2`, which has no score log. Its files re-score to exactly those values.
- Resolved: paper Table 6 lists WR/L/DTD 0.85 for the 5-token TurkCorpus row. The Table 2 and 5 scores (43.00/64.42/5.63) use 0.75. The human evaluation (Table 3) rated outputs made at 0.85 (43.04/75.11/6.54): all 100 rated sentences equal `exp_1601582154282931_best/outputs.3/dbd23931...`. So the paper reports two settings of one model.
- The original-pipeline scores for the released models are below the README because they are other decoding passes.
  Re-scored with the pinned easse, those files give their logged values exactly (`docs/RESULTS.md`).

## Hyperparameter and token searches (Optuna)

Source folder: `TS_backup/TS_T5.2/experiments/tuning_logs/old/` (three SQLite studies). We read copies, never the files in place.

| Study | Objective | Trials | Result |
|---|---|---|---|
| `Tokens_study.db` | maximise SARI over C, L, WR, DTD in 0.20-1.50, step 0.05 | 239 complete, 65 failed, 4 unfinished; 2021-05-24 to 2021-05-25 | best 43.61 at C 0.90, L 0.80, WR 0.55, DTD 0.80 (trial 83). See `docs/FINDINGS.md`, finding 2. |
| `TS_T5_study.db` | minimise `val_loss` over learning rate (1e-6 to 1e-3), batch size (6, 12, 18, 32), epochs (3, 5, 8), adam epsilon | 15 complete, 9 pruned, 2 failed; 2021-05-24 to 2021-05-25 | best `val_loss` 0.8159 at lr 2.8e-6, batch 18, 3 epochs, epsilon 1.2e-7 (trial 19) |
| `TS_T5_study.1.db` | same code, smaller | database file is damaged (`database disk image is malformed`) | not read |

The hyperparameter study used the misaligned May 2021 validation (finding 1), so its `val_loss` is not a SARI proxy.
The released models used lr 3e-4 and 5 epochs, not the values from this study, and they were trained in 2020, before it.

## Hardware and run times

See `docs/REPRODUCE.md`, section "Original hardware and run times".
