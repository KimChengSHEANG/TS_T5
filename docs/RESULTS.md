# Results

Scores for the two released TS-T5 models on the TurkCorpus and ASSET test sets (359 sentences each).
The README numbers are the canonical ones. They are re-scored here from the files in `system_outputs/`.

## How the scores are made

- Command: `python scripts/make_docs_tables.py scores` (reads `system_outputs/`, scores each file with `ts_t5.evaluate.score_file`).
- Metric: `easse` `evaluate_system_output(test_set="<name>_test", lowercase=True)`, pinned to commit
  `6a4352ec299ed03fda8ee45445ca43d9c7673e89` (see `docs/REPRODUCE.md`).
- Every file was checked for 359 lines before scoring. The ASSET source file has no trailing newline, so `wc -l` says 358.
- The five baseline files in `system_outputs/` hold TurkCorpus outputs only.

## Scores (SARI / BLEU / FKGL)

Source: output of `python scripts/make_docs_tables.py scores`, run on the files named in the last column.

| System | Test set | SARI | BLEU | FKGL | File |
| --- | --- | --- | --- | --- | --- |
| ACCESS | turkcorpus | 41.38 | 76.36 | 7.29 | `system_outputs/ACCESS.txt` |
| DMASS-DCSS | turkcorpus | 39.92 | 73.29 | 7.73 | `system_outputs/DMASS-DCSS.txt` |
| Dress-Ls | turkcorpus | 36.97 | 81.08 | 7.66 | `system_outputs/Dress-Ls.txt` |
| MUSS(Bart+ACCESS) | turkcorpus | 42.24 | 76.33 | 6.58 | `system_outputs/MUSS(Bart+ACCESS).txt` |
| yats | turkcorpus | 37.39 | 74.87 | 7.67 | `system_outputs/yats.txt` |
| TS-T5 4-token | turkcorpus | 43.50 | 67.26 | 6.17 | `system_outputs/TS_T5_4tokens/d859b9c4aab9f7a9c719167728399278_turkcorpus.test.txt` |
| TS-T5 4-token | asset | 45.05 | 72.24 | 6.33 | `system_outputs/TS_T5_4tokens/d859b9c4aab9f7a9c719167728399278_asset.test.txt` |
| TS-T5 5-token | turkcorpus | 43.34 | 65.54 | 5.55 | `system_outputs/TS_T5_5tokens/267a9c5912b4c0e37bf8bf7260f7576d_turkcorpus.test.txt` |
| TS-T5 5-token | asset | 45.26 | 72.19 | 5.92 | `system_outputs/TS_T5_5tokens/267a9c5912b4c0e37bf8bf7260f7576d_asset.test.txt` |

Default control tokens (the file name is the md5 hash of the token dict, see `docs/ABLATIONS.md`):

- 4-token: `C 0.95, L 0.75, WR 0.75, DTD 0.75` (hash `d859b9c4...`).
- 5-token: `W 1.05` plus the same four (hash `267a9c59...`).

Both models are read from the local release folders `experiments/hf/{4,5}tokens`. The same files are on Hugging Face as
`kcsheang/ts-t5-base-4tokens` and `kcsheang/ts-t5-base-5tokens`.

## Do the released weights give these numbers?

Yes. The released weights are the Google-Drive (2020) runs, decoded at `max_seq_length` 256. Fresh generation on CPU gives
SARI within 0.03 of the table on all four cells. Details and hashes: `docs/REPRODUCE.md`, section "Reproduction gate".

| Model | Test set | Fresh run SARI / BLEU / FKGL | Table above |
|---|---|---|---|
| 4-token | TurkCorpus | 43.49 / 67.23 / 6.17 | 43.50 / 67.26 / 6.17 |
| 4-token | ASSET | 45.08 / 72.37 / 6.31 | 45.05 / 72.24 / 6.33 |
| 5-token | TurkCorpus | 43.33 / 65.53 / 5.55 | 43.34 / 65.54 / 5.55 |
| 5-token | ASSET | 45.27 / 72.20 / 5.91 | 45.26 / 72.19 / 5.92 |

Source: `docs/REPRODUCE.md` (gate table, `ts-t5 evaluate --device cpu`, beam 8).

## Paper numbers are lower than the README

The README numbers (canonical here) are all higher than Tables 2 and 5 of the INLG 2021 paper
(Sheang and Saggion, "Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer").
Paper values: `docs/REPRODUCE.md`, section "Paper vs README".

| Model | Test set | Paper SARI / BLEU / FKGL | README SARI / BLEU / FKGL |
|---|---|---|---|
| 4-token | TurkCorpus | 43.31 / 66.23 / 6.17 | 43.50 / 67.26 / 6.17 |
| 4-token | ASSET | 44.91 / 71.96 / 6.32 | 45.05 / 72.24 / 6.33 |
| 5-token | TurkCorpus | 43.00 / 64.42 / 5.63 | 43.34 / 65.54 / 5.55 |
| 5-token | ASSET | 45.04 / 71.21 / 5.88 | 45.26 / 72.19 / 5.92 |

What we can say, and what we cannot:

- The paper values are also in the 2020-21 score file `TS_T5/results/for_ablations.txt` (see `docs/ABLATIONS.md`).
  The 4-token rows with `C0.95 L0.75 WR0.75 DTD0.75` give 43.31/66.23/6.17 (Turk) and 44.91/71.96/6.32 (ASSET).
  The 5-token rows with `W1.05 C0.95 L0.75 WR0.75 DTD0.75` give 43.00/64.42/5.63 and 45.04/71.21/5.88.
- The weights behind those rows are the released weights: `pytorch_model.bin` in the D8 backup folders
  `exp_1607800772213084_C_WR_L_DTD` and `exp_1601582154282931_best` has the same sha256 as the Drive zips
  (`docs/REPRODUCE.md`, section "Weight hashes").
- So the gap between paper and README probably comes from the scoring or decoding side, not from the weights.
  We re-scored the README outputs with the pinned easse and got the README numbers, so easse (not the model) is
  at most one part of the story. We could not find which setting produced the lower paper numbers.
- The paper's 5-token TurkCorpus row came from per-test-set searched tokens (Table 6: `WR/L/DTD 0.85`; see `docs/REPRODUCE.md`, "Paper vs README"). The README, `system_outputs/` and the 43.00 row in `for_ablations.txt` all use 0.75.
  Confirmed conflict: paper Table 6 lists WR/L/DTD 0.85 for the 5-token TurkCorpus row, but the Table 5 numbers (43.00/64.42/5.63) match the 0.75 row of `for_ablations.txt` (the 0.85 row reads 42.98/75.15/6.58). We do not know which of the two the paper intended. We left the
  paper text alone.

## Five example simplifications

Sources are the test files `resources/datasets/turkcorpus/turkcorpus.test.complex` (TurkCorpus) and
`resources/datasets/asset/asset.test.orig` (ASSET). Outputs come from the gate run
`experiments/out_drive256/{4,5}tokens/<hash>_<test set>.test.txt` (untracked). These lines are identical in the tracked
`system_outputs/TS_T5_{4,5}tokens/` files. Detokenised text is as the models print it. Line numbers start at 1.

**1. TurkCorpus line 3.**
- Source: The Great Dark Spot is thought to represent a hole in the methane cloud deck of Neptune .
- 4-token: The Great Dark Spot is thought to be in the methane cloud deck of the planet Neptune.
- 5-token: The Great Dark Spot is thought to be a hole in the cloud deck of the planet Neptune.

**2. TurkCorpus line 45.** Both models swap a hard phrase for a plain one.
- Source: Formal minor planet designations are number-name combinations overseen by the Minor Planet Center , a branch of the IAU .
- 4-token: Minor planet designations are given by the Minor Planet Center, a part of the International Astronomical Union.
- 5-token: Minor planet designations are given by the Minor Planet Center, a part of the International Astronomical Union ( IAU ).

**3. TurkCorpus line 336.** The 4-token model drops "five". The 5-token model keeps it.
- Source: A majority of South Indians speak one of the five Dravidian languages — Kannada , Malayalam , Tamil , Telugu and Tulu .
- 4-token: Most South Indians speak one of the Dravidian languages. These are Kannada, Malayalam, Tamil, Telugu and Tulu.
- 5-token: Most South Indians speak one of the five Dravidian languages. These are Kannada, Malayalam, Tamil, Telugu, and Tulu.

**4. ASSET line 9.** A split plus a rewrite.
- Source: Following the drummers are dancers, who often play the sogo (a tiny drum that makes almost no sound) and tend to have more elaborate — even acrobatic — choreography.
- 4-token: After the drummers are dancers, who often play the sogo ( a tiny drum that makes almost no sound ). Dancers often have choreography that is more complicated.
- 5-token: After the drummers are dancers, who often play the sogo ( a tiny drum that makes almost no sound ). Dancers often do more elaborate choreography, such as acrobatics.

**5. ASSET line 25.** Light edit: "emulated" becomes "used".
- Source: Nevertheless, Tagore emulated numerous styles, including craftwork from northern New Ireland, Haida carvings from the west coast of Canada (British Columbia), and woodcuts by Max Pechstein.
- 4-token: Tagore used many different styles, including craftwork from northern New Ireland, Haida carvings from the west coast of Canada ( British Columbia ) and woodcuts by Max Pechstein.
- 5-token: Tagore used many different styles, such as craftwork from northern New Ireland, Haida carvings from the west coast of Canada ( British Columbia ) and woodcuts by Max Pechstein.

Cases where the output is wrong are in `docs/FINDINGS.md`, section "Failure cases".

## Figures

Compression histograms from the 2021 `plots/` scripts (D8 data, TurkCorpus validation split) are in `docs/figures/`.
They are described in `docs/FINDINGS.md`, section "What the length tokens do".
