# Human evaluation

Crowd ratings of simplifications from four systems (YATS, DMASS-DCSS, MUSS, TS-T5), collected on Amazon Mechanical Turk. The ratings are published anonymised in `human_evaluation/`.

## Protocol

Source: the evaluation notes and the HIT designs in `human_evaluation/HIT_designs/`.

- Raters judge each simplified sentence on a 5-point scale (1 = a lot worse, 3 = about the same as the original, 5 = a lot better) for three aspects:
  - Fluency (grammaticality): is it well-formed and grammatically correct?
  - Simplicity: is it simpler than the original sentence?
  - Adequacy (meaning preservation): does it preserve the meaning of the original sentence?
- One HIT shows 5 source sentences, each with 4 simplifications from the 4 systems (system order is shuffled per HIT, and the system is not shown). The final HIT is `HIT_designs/mturk_rating_5sents.html`.
- 100 source sentences per study (20 HITs), 5 raters per HIT, so 500 ratings per system and aspect.
- Raters had to be English native speakers, 18 or older, and had to accept an information sheet and consent form. Data is used for research only.
- Sentences: the paper (INLG 2021, Table 3) says the 100 sentences were drawn at random from the TurkCorpus test set. The D8 evaluation notes also list candidate sources with sentence counts (Zhao et al. 2020: 200; Kriz et al. 2019: 200; Jiang et al. 2020: 100; Zhang and Lapata 2017: 100; Mallinson and Lapata 2019: 100, 50 from each model). Those are the authors' working notes, not a description of the final sample.

## Files

| Path | Content |
| --- | --- |
| `human_evaluation/mturk_batches/5tokens_batch_v{1,2,3}.csv` | Raw ratings, 3 batches (2 + 5 + 13 HITs), anonymised |
| `human_evaluation/mturk_batches/4tokens_batch_4442003.csv` | Raw ratings, one batch of 20 HITs, anonymised |
| `human_evaluation/extracted_ratings/{4tokens,5tokens}/<system>.csv` | Per-system table: sentence, 5 ratings per aspect and their average (from the authors' xlsx) |
| `human_evaluation/extracted_ratings/*/complex.txt` | The 100 source sentences |
| `human_evaluation/HIT_designs/` | HIT HTML (requester name and email removed) |

Column meaning in `mturk_batches/`: `HITId`, `AssignmentId`, `WorkerId` are salted SHA-256 hashes (12 hex characters). The 4-token batch has no `WorkerId` column; in the 5-token batches the same rater keeps the same `WorkerId`. `Input.*` hold the sentences and the system index (`simp_sent_id_*`: 0 YATS, 1 DMASS-DCSS, 2 MUSS, 3 TS-T5). `Answer.fluency_jk`, `Answer.simplicity_jk`, `Answer.adequacy_jk` are the ratings for sentence `j`, system slot `k`. Everything else (timestamps, approval and rejection fields, free-text suggestions, worker approval rates) was dropped.

## Aggregate scores

Mean rating over 500 ratings (100 sentences x 5 raters). Command: `python scripts/anonymise_human_eval.py scores`.

From `human_evaluation/mturk_batches/5tokens_batch_v*.csv` (TS-T5 here is the 5-token model; this is the paper's Table 3):

| System | Fluency | Simplicity | Adequacy |
| --- | --- | --- | --- |
| YATS | 4.032 | 3.620 | 3.924 |
| DMASS-DCSS | 3.848 | 3.706 | 3.482 |
| MUSS | 4.410 | 4.022 | 4.130 |
| TS-T5 | 4.308 | 3.990 | 4.180 |

From `human_evaluation/mturk_batches/4tokens_batch_4442003.csv` (not in the paper; TS-T5 here is the 4-token model):

| System | Fluency | Simplicity | Adequacy |
| --- | --- | --- | --- |
| YATS | 3.910 | 3.608 | 3.902 |
| DMASS-DCSS | 3.620 | 3.486 | 3.350 |
| MUSS | 4.230 | 3.882 | 3.954 |
| TS-T5 | 4.114 | 3.692 | 3.860 |

Cross-check from `human_evaluation/extracted_ratings/5tokens/*.csv`: identical to the first table. From `extracted_ratings/4tokens/*.csv`: within 0.015 of the second table (largest gaps: YATS fluency 3.925 against 3.910 and YATS adequacy 3.917 against 3.902; TS-T5 4.111/3.693/3.851 against 4.114/3.692/3.860).

Comparison with the authors' summary table (`results/5tokens/final data/readme.md` in the source folder, not the paper itself): it lists YATS 4.032/3.62/3.924, DMASS-DCSS 3.848/3.706/3.482, MUSS 4.41/4.022/4.13 and "Our Model" 4.308/3.99/4.18. These match the `5tokens` batches to all digits. They do not match the `4tokens` batch (TS-T5 4.114/3.692/3.860).

Caveats:

- The paper (Table 3) reports "Our Model" as "T5-base+All Tokens", the 5-token model, and its numbers match the `5tokens` batches (the paper shows 4.30 where this doc shows 4.308, a truncation). The `4tokens` batch is an extra run that is not in the paper. The tables below label TS-T5 accordingly. Which model each batch rated is inferred from the folder names.
- The paper names the MUSS row "BART+ACCESS" (Martin et al. 2020a). Same system.
- The D8 evaluation notes contain no numeric table (it is an image); the comparison with the paper uses its Table 3.
- The 4-token extracted CSVs carry mojibake from the source xlsx (for example `H√§rtel`). It is kept as found.

## Provenance and privacy

The raw MTurk exports under `Human evaluation/results/raw from mTurk/` predate this release and are untouched. They contain the original worker IDs. Use `human_evaluation/` for anything public. The salt used for hashing is kept outside the repo, so the hashes cannot be reversed by a dictionary attack on worker IDs. The 5-token rows can still be matched to the raw exports by their answers, so they are pseudonymised, not anonymous. The 4-token batch, which has no raw export in this repo, omits `WorkerId` so that it cannot be joined to them through a rater ID.

To rebuild: `TS_T5_SALT="$(cat ~/.config/ts_t5/salt)" python scripts/anonymise_human_eval.py build "<path to TS_T5 Human evaluation>"`. A new salt gives different hashes.
