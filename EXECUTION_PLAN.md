# DeepATM Reconstruction — Execution Plan

Build order, data sourcing, and per-milestone acceptance criteria.

Source of truth for the architecture: Lee et al., *Cell* 188:5081–5099 (2025),
STAR★Methods §"Deep learning dataset and feature engineering", §"Model
architecture", §"Model training", §"Performance evaluation", §"Predicting the
effects of unevaluated ATM variants", plus Figure 6A. The spec is transcribed
verbatim in [§6](#6-frozen-spec-from-starmethods) and has been checked against
the PDF.

> **Revision note (2026-08-17).** An earlier version of this plan excluded the
> supplementary tables and pivoted training to BRCA1/BAP1 SGE data. That premise
> was wrong: `mmc1.xlsx` is available locally, and it reproduces the paper's
> dataset numbers exactly (§1). The plan is now ATM-first. The public-source
> download layer survives, because it is still needed for the 11 missing
> auxiliary scores, real Cα coordinates, and the ≥2★ ClinVar subset — just not
> for labels.

---

## 0. What this project is

A from-scratch reimplementation of DeepATM from the published methods, trained
on the paper's own function scores, evaluated against the paper's own reported
numbers. The authors' code is not public ("available upon request", RRID: N/A),
so this is a reconstruction, not a re-run.

**Success = reproducing the paper's three headline numbers**, within the
tolerance that unstated hyperparameters allow:

| Quantity | Paper | Source |
|---|---|---|
| 5-fold CV Pearson r (with structure) | **≈ 0.61** | Fig. 6B / Fig. S7A |
| eDA ↔ function score correlation (n=23,092) | **r = 0.70** | Fig. 6C |
| auROC, ClinVar ≥1★ missense test set (n=116) | **0.95** | Fig. 6D |

Secondary targets: the paper's ablation (a transformer *without* structural
coordinates scores lower, p = 0.032) and its baseline (a random forest on
missense reaches 0.55 vs DeepATM's 0.61). Both are cheap to reproduce and are
the sharpest evidence that the reconstruction is faithful — a model that hits
r ≈ 0.61 but shows no structure ablation gap has probably learned something
other than what DeepATM learned.

---

## 1. The data situation, verified

`mmc1.xlsx` (Table S1, "Combined scores for all ATM SNVs") is the label set.
Every count below was computed directly from the file, not assumed:

| Quantity | From `mmc1.xlsx` | Paper | |
|---|---|---|---|
| Total rows | 29,373 | — | |
| Measured, all consequences (`DeepATM_predicted == "No"`) | 24,534 | 24,534 | ✅ |
| Measured **coding** (missense + synonymous + nonsense) | **23,092** | 23,092 | ✅ |
| DeepATM-predicted (`== "Yes"`) | **4,421** | 4,421 | ✅ |
| ClinVar test set: missense P/LP/B/LB | **116** | 116 | ✅ |
| Training missense (after position exclusion) | **16,275** | 16,275 | ✅ |
| Training nonsense (after dropping stop-codon position) | **1,183** | 1,183 | ✅ |
| Training synonymous | 5,031 | 4,395 | ❌ Δ=636 |
| Training total | **22,489** | 21,853 | ❌ Δ=636, all of it synonymous |

Two consequences of this that change the build:

**The test set needs no ClinVar download.** The `ClinVar_classification` column
in Table S1 is *already* ≥1★-filtered by the authors — selecting missense rows
whose classification is Pathogenic / Likely pathogenic / Pathogenic-Likely
pathogenic / Benign / Likely benign / Benign-Likely benign yields exactly 116
variants at 103 distinct amino-acid positions, and excluding those 103 positions
from the measured set leaves exactly 16,275 missense. The paper's split is
byte-for-byte reconstructible. The ≥2★ subset (n=68) is *not* — Table S1 carries
no review-status column — so that one test still requires `variant_summary.txt.gz`.

**The synonymous count is the one open discrepancy**, and which number it takes
depends on a genuine ambiguity in the paper's own text. The paper says the
training set excludes "all evaluated variants that shared amino acid positions
with the test set", but its published counts contradict that reading:

| Exclusion applied to | missense | nonsense | synonymous |
|---|---|---|---|
| all coding variants | 16,275 ✅ | 1,148 ❌ (paper 1,183) | 4,857 |
| **missense only** (default) | **16,275 ✅** | **1,183 ✅** | **5,031** |

Nonsense reproduces exactly only when the rule is *not* applied to it, so
missense-only is the reading consistent with the paper's own numbers, and it is
`data_prep.py`'s default. It leaves synonymous at 5,031 against the paper's
4,395 — the full 636 unexplained. Neither stop-codon exclusion, deduplication
on amino-acid change, nor ClinVar status produces 4,395; all three were tested.

`--position-exclusion coding` takes the stricter, lower-leakage reading (a
synonymous variant at a test position still trains that position's embedding
against a label) at the cost of nonsense no longer matching. Use it for claims
about generalisation, the default for reproduction. Either way this is a known,
documented difference, not a bug to chase.

**On redistribution.** The supplement is Elsevier/Cell Press copyright. Using a
local copy for a study reconstruction is ordinary; committing it to a public
repo is not. `.gitignore` already excludes `data/raw/*`. Keep it that way, and
keep `README.md`'s "download it yourself" instructions — they are what make the
repo reproducible without redistributing anything.

---

## 2. External data — what still has to be downloaded

Labels come from `mmc1.xlsx`. Everything below is free and programmatically
fetchable. None of it is redistributed; `src/sources/` downloads into
`data/raw/` on first run and caches.

### 2.1 The 16 auxiliary scores

Table S1 ships **5 of 16**: `CADD.phred`, `boostDM_score`, `EVE_scores_ASM`,
`REVEL`, `AlphaMissense`. Missingness within those five is substantial —
23% (CADD) to 39% (EVE) of rows — because they are undefined for synonymous and
intronic variants. The remaining 11 must be fetched:

| Source | Covers | How |
|---|---|---|
| **dbNSFP v5.1** | SIFT, FATHMM, MutationTaster, LRT, DANN, PolyPhen-2 HVAR, PROVEAN, phyloP100way, GERP++, ESM1b (+ CADD, REVEL, AlphaMissense as cross-checks) | Register at <https://www.dbnsfp.org/download> (free, CC BY-NC-ND, ~50 GB). Tabix-slice `chr11:108,222,500–108,369,100` (GRCh38) only — do not load the whole file. |
| **SpliceAI** | SpliceAI | Illumina precomputed VCFs (free academic), or the Broad SpliceAI-lookup API for small sets |

EVE and BoostDM are already in Table S1 for ATM, so no separate fetch is needed
on this track.

Every column keeps a paired `*_is_missing` indicator. Do **not** mean-impute
without the flag — a mean-imputed SIFT score for a synonymous variant is a
fabricated value the model will happily fit. Column order stays fixed at 16 so
`N_AUX_SCORES` matches the paper.

### 2.2 Cα coordinates

The paper used **AlphaFold 3**, which is not deposited. Verified alternatives:

1. **AlphaFold DB has no ATM entry.** `AF-Q13315-F1-model_v4.cif` → HTTP 404;
   the prediction API returns `{}`. ATM (3,056 aa) is past AFDB's length
   cut-off, and there are no fragment entries. Do not plan around AFDB.
2. **Use PDB 7SID or 8OXQ.** Both are cryo-EM ATM dimers with SIFTS coverage of
   UniProt 1–3056. For 7SID chain A, SIFTS reports `unp_end 3056 ↔
   auth_residue 3056` — the numbering **is** 1:1, so the existing
   `auth_seq_id`-as-position assumption happens to be correct here. Verify it
   per structure via `https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/<id>`
   rather than trusting it.
3. **`PDB_IDS = ["8OXO", "7SID"]` in `features.py:83` is wrong.** 8OXO is a
   **12-residue synthetic peptide**, not ATM. It contributes 12 junk coordinates
   to the track. The intended entry is **8OXQ** (3,184-residue construct, ATM).
4. **7SID also contains nibrin (NBN, O60934) in chains B and D.** The current
   loop walks every chain in the first model, so NBN residues 727–754 can land
   in the ATM coordinate track. Filter to the ATM chain explicitly.
5. Cryo-EM structures have unmodelled loops even at SIFTS coverage 1.0. Residues
   with no observed Cα get **masked**, and the coordinate MLP sees a learned
   "missing" vector. **Delete `synthetic_backbone()`** (`features.py:126`) — a
   fabricated helix silently teaches the model a fake geometry, and worse, the
   current code caches it to `ca_coordinates.npy` on the first failed fetch and
   never re-tries.

### 2.3 ClinVar ≥2★ subset (secondary test only) — **done**

Implemented in `src/clinvar.py`; run `python -m src.clinvar`.

Source is the **GRCh38 release VCF**, not `variant_summary.txt.gz` as this plan
originally specified. Both carry review status, but the VCF's coordinates are
already GRCh38 VCF-normalised, which makes the join to Table S1's
`hg38_pos`/`Ref`/`Alt` exact rather than assembly-filtered-then-hoped-for; it is
roughly a third of the size; and `##fileDate` pins the release without a
separate lookup. Only the ~150 kb ATM locus is parsed.

Result, against release **2026-08-08**:

| | |
|---|---|
| ≥1★ test variants matched in ClinVar | **116 / 116** |
| ≥2★ and still classified P/LP or B/LB | **70** (paper: 68) |
| of which P/LP · B/LB | 45 · 25 |
| star histogram over the 116 | 0★:1 · 1★:45 · 2★:50 · 3★:20 |
| reclassified since Table S1's snapshot | 0 |

A clean 116/116 match is itself a check on the whole reconstruction: it
confirms the test set derived from Table S1's `ClinVar_classification` column
really is the ClinVar-sourced set the paper describes. n=70 vs 68 is the
expected direction and magnitude of two years of ClinVar growth (D5).

The release URL, date and SHA-256 are written to
`data/processed/clinvar_release.json` and echoed into `outputs/metrics.json`,
so any reported ≥2★ number carries its provenance.

**Size-matching is deliberately not reported here.** Drawing 68 of 70 leaves
almost nothing to vary, so the resulting interval measures which two rows were
dropped, not sampling uncertainty — printing it next to a real bootstrap CI
would invite exactly the wrong comparison. The code does it only when the
subset exceeds the target by >10%.

### 2.4 Not needed

UniProt domain spans (the paper's own table is authoritative for ATM and is
already in `features.py:44`, verified against the PDF), gnomAD (`GnomAD_all_AF`
is in Table S1), and MaveDB (no ATM score set exists; irrelevant now that labels
come from S1).

---

## 3. Milestones

Each ends with a runnable command and a check. Do not start the next until the
check passes.

### M0 — Environment
`torch` is not currently installed. Everything except `data_prep` is unrunnable
until it is.
✅ `python -c "import torch; print(torch.__version__)"` and `pytest tests/` both pass.

### M1 — Splits that match the paper
Extend `data_prep.py` to emit the paper's actual split, not just
measured/predicted:
- `test.csv` — the 116 ClinVar missense P/LP/B/LB rows (§1);
- `train.csv` — measured coding rows, excluding the 103 test positions and the
  9 rows at position 3057 (stop codon);
- `predict.csv` — the 4,421 `DeepATM_predicted == "Yes"` rows.

Rename the target column on load: Table S1's `Combined_score` is the **function
score** for measured rows and the paper's **own eDA output** for predicted rows.
Training on the wrong subset would be self-supervision on the model being
reconstructed. Make that impossible to do by accident.
✅ Row counts print as 116 / 22,489 (16,275 + 5,031 + 1,183, our synonymous
count) / 4,421, and the missense and nonsense figures match the paper exactly.

### M2 — Features
Rewrite `features.py`:
- real Cα coordinates: 8OXQ or 7SID, ATM chain only, SIFTS-verified numbering,
  per-residue missing mask, `synthetic_backbone()` deleted (§2.2);
- 16-score matrix with per-column missingness flags (§2.1);
- **fix the leakage**: `ScoreImputer.transform` (`features.py:213`) recomputes
  mean and σ from whatever frame it is handed, so train and val get different
  normalisation. Fit on the training fold only, store the constants, apply them
  everywhere. Same for `.fit(df)` at `train.py:201`, which currently fits on all
  folds before splitting.
- keep `arcsinh_transform` — verified verbatim against the paper.
✅ `tests/test_features.py`: score matrix is (n, 16); no NaN; train and val share
one imputer instance; the coordinate array's residue-to-residue distance
distribution looks like a real protein, not a helix.

### M3 — Model
`model.py` is close to correct. Three changes:
- **Add positional encoding.** `nn.TransformerEncoder` has none, and
  `encode_sequence` (`model.py:104`) sums only aa + domain + coordinate
  embeddings. The coordinate branch does inject position-dependent signal, so
  the encoder is not fully permutation-equivariant — but it has no ordinal sense
  of the sequence. The paper does not specify this either way; add sinusoidal or
  learned embeddings and log it as a deviation (D3).
- **Make L=3,056 tractable.** Attention alone at B=20, L=3056, 8 heads, fp32 is
  ≈6 GB *per layer* forward, before activations or backward. Route through
  `F.scaled_dot_product_attention` (flash/mem-efficient backend) and use fp16
  under AMP. Keep the CPU windowed mode for smoke runs only, and report the
  window size with every result — a windowed model is not the paper's model.
- Assert `embed_dim % n_heads == 0` and `aux_scores.shape[1] == 16`. Drop the
  unused `n_residues` constructor argument.
✅ `python -m src.model` prints the parameter count and completes a forward pass
at full L on the target device.

### M4 — Training
Implement §6 literally. Three things the current code gets wrong:
- **The 20% LR decay is a no-op.** `train.py:154` multiplies
  `param_groups[...]["lr"]` by 0.8, but `CosineAnnealingWarmRestarts.step()`
  recomputes lr from `base_lrs` on the next call and overwrites it. Decay
  `scheduler.base_lrs` instead, and trigger on actual restart boundaries
  (10, 30, 70, 150 for `T_0=10, T_mult=2`), not on `epoch % 10 == 0`.
- **`load_reference_sequence` is called on the training subset**
  (`train.py:198`). Positions absent from that subset silently become `"X"` —
  catastrophic under `--max-rows`. Build the wild-type sequence once from the
  full Table S1 and cache it.
- `torch.cuda.amp.GradScaler` (`train.py:141`) is deprecated on torch ≥ 2.4;
  use `torch.amp.GradScaler("cuda")`.

The batch sampler is correct as written — 20 × (0.90, 0.05, 0.05) → 18/1/1.
✅ 5 fold checkpoints in `checkpoints/`; LR curve shows four restarts with
visibly decaying peaks; a 2-epoch smoke run completes on CPU.

### M5 — Evaluation
`evaluate.py` needs a rewrite, not a patch. Four defects, each of which alone
invalidates its output:
1. **It evaluates on the training set.** It loads all of `measured.csv` and
   reports correlation on it (`evaluate.py:79`). These are in-sample numbers.
   Report per-fold correlations on held-out folds, and take the median across
   folds as the paper does.
2. **The auROC sign is inverted.** Low function score = loss of fitness =
   pathogenic. Passing `preds` directly to `roc_auc_score` with pathogenic as
   the positive class yields ≈ 1 − AUC. Use `-preds`.
3. **"Conflicting classifications of pathogenicity" is scored as pathogenic.**
   The substring test at `evaluate.py:99` matches "pathogenicity" — 422 rows in
   the measured set. Match against an explicit label set.
4. **No star filter.** For the ≥2★ test the ClinVar join from §2.3 is required.

Then: ensemble the 5 checkpoints, bootstrap 1,000 resamples for CIs, and compare
against AlphaMissense, ESM1b, phyloP, PROVEAN.
⚠️ Those four baselines are also model *inputs*. Report the comparison as the
paper does, but state plainly that it measures "does the transformer add anything
on top of its own features", not independent superiority.
✅ `outputs/metrics.json` with per-fold r/ρ, ensemble auROC + CI, baseline auROCs.
Compare against the §0 targets.

### M6 — Ablation and baseline
The two checks that most cheaply falsify a bad reconstruction:
- retrain with the coordinate branch removed → expect a drop, p ≈ 0.032;
- random forest on missense only, same 16 scores → expect ≈ 0.55 vs ≈ 0.61.
✅ Both reported in `outputs/metrics.json` alongside the paper's values.

### M7 — eDA scores
Ensemble-predict the 4,421 unevaluated SNVs, then apply the paper's calibration:
rank-align raw predictions for the 23,092 measured variants to their function
scores, fit a **generalized additive model** (`pygam`) to that relationship, and
map unmeasured predictions through it. Classify at the published cutoffs
(−1.360 = 5th percentile of synonymous; −0.912 = Youden's index for nonsense vs
synonymous) — both verified against the PDF.

Two fixes to `predict.py`:
- `rank_align` (`predict.py:47`) uses in-sample ensemble predictions on the
  measured set. Use **out-of-fold** predictions, or the calibration inherits the
  ensemble's memorisation of its own training data.
- Replace the `searchsorted` step function with the GAM the paper specifies.

✅ `outputs/eda_scores_reconstruction.csv`. The headline check is already wired
in at `predict.py:100`: correlate against the paper's published eDA scores,
which are in Table S1's `Combined_score` for these rows and were never used in
training. **This is the single best validation the project has** — a direct,
variant-level comparison against the original model's output.

### M8 — Cross-gene generalisation (optional)
Only after M1–M7 land. Train on public SGE data (BRCA1
`urn:mavedb:00000097-0-2`, n=3,893, CC0; BAP1 `urn:mavedb:00000662-0-1`,
n=18,108, CC BY 4.0) with a gene-identity embedding, and test transfer to ATM.
This answers "is the architecture general or ATM-specific" — a genuine research
question, but a different one from reproducing the paper. Do not let it displace
the primary track.

---

## 4. Known defects in the current code

Consolidated. Everything below was read in the source, and the data-dependent
ones were checked against `mmc1.xlsx`.

| Location | Issue | Severity |
|---|---|---|
| `evaluate.py:79` | Reports correlation on the full training set — in-sample metrics | **Invalidating** |
| `evaluate.py:103` | auROC sign inverted; low score = pathogenic | **Invalidating** |
| `features.py:213`, `train.py:201` | Normalisation statistics refit per call and fit before folding → leakage and train/val skew | **Invalidating** |
| `features.py:126,141` | `synthetic_backbone()` fabricates geometry, and caches it permanently on first fetch failure | **Invalidating** |
| `features.py:83` | `8OXO` is a 12-residue peptide, not ATM; should be `8OXQ` | High |
| `features.py:111` | Iterates all chains; 7SID contains NBN, which contaminates the ATM track | High |
| `train.py:154` | The 20%-per-restart LR decay is overwritten by the scheduler — currently a no-op | High |
| `train.py:198` | `load_reference_sequence` built from the training subset; unseen positions become `"X"` | High |
| `evaluate.py:99` | "Conflicting classifications of pathogenicity" counted as pathogenic (422 rows) | Medium |
| — | The paper's position-exclusion split is not implemented anywhere; no ClinVar test set exists | Medium |
| `predict.py:47` | Rank calibration uses in-sample predictions; should be out-of-fold | Medium |
| `model.py:104` | No positional encoding | Medium |
| `dataset.py:74` | Trains on `Combined_score`, which is the paper's own eDA output for predicted rows | Medium |
| `data_prep.py:106` | Drops 9 rows at position 3057, so `measured.csv` is 23,083 not 23,092. Correct behaviour (stop-codon position), wrong stage — it should happen when building the training set, and the README's 23,092 should be explained | Low |
| `train.py:141` | `torch.cuda.amp.GradScaler` deprecated on torch ≥ 2.4 | Low |
| `model.py:66` | `n_residues` argument stored but never used | Low |

---

## 5. Environment

Add to `requirements.txt`: `pyarrow` (parquet), `pysam` or `tabix` (dbNSFP
slicing), `pygam` (M7 calibration), `gemmi` (fast mmCIF + SIFTS; replaces the
Biopython `MMCIFParser` path), `pytest` (the test suite has no declared runner),
`wandb` (optional; the paper used v0.12.15).

Paper's stack, for reference: Python 3.10.15, PyTorch 1.11.0, pandas 1.3.5,
numpy 1.21.5, scikit-learn 1.0.2, SciPy 1.7.3, Biopython 1.81.

---

## 6. Frozen spec from STAR★Methods

Transcribed from the PDF and checked. Do not "improve" these — deviations go in
the register below.

**Target transform.** `y = asinh((function_score + 0.912) / 2)`

**Domains** (verified identical to `features.py:44`):
TAN 1–166 · FAT 1940–2566 · PI3/4 Kinase 2686–2998 · FATC 3024–3056

**Architecture.**
- Amino-acid embedding: 64-d, randomly initialised, one per residue.
- Domain embedding: separate embedding layer, per residue.
- Coordinate embedding: MLP over Cα (x,y,z) from AlphaFold 3, "integrated with"
  the amino-acid and domain embeddings. (The paper does not say *how* they are
  combined; summation is this reconstruction's choice — see D7.)
- Encoder: **2** Transformer encoder layers × **8** attention heads.
- Head: encoder output **at the mutated position** ‖ 16 precomputed scores →
  Linear(128) → ReLU → Linear(1).

**Training.** AdamW, lr 1e-3, weight decay 1e-2. Cosine annealing with periodic
restarts: initial cycle 10 epochs; after each restart LR reduced by 20% and cycle
length doubled. Max 150 epochs, early stopping patience 20. Batch size **20**,
dynamically resampled each batch to 90% missense / 5% synonymous / 5% nonsense.
MSE loss. Automatic mixed precision. Gradient clipping.

**Splits.** Test set = all missense variants classified P/LP/B/LB at ≥1★ in
ClinVar (n=116), *irrespective of whether they were experimentally evaluated* —
so some test variants have no function score and are scored on ClinVar label
alone. Training set = all remaining evaluated variants after excluding every
variant sharing an amino-acid position with the test set: 16,275 missense /
4,395 synonymous / 1,183 nonsense. Mutations at stop-codon positions excluded.

**Evaluation.** 5-fold CV, random splits, Pearson + Spearman. Checkpoint on
val-loss improvement; ensemble the 5 best. auROC on ClinVar ≥1★ (n=116) and ≥2★
(n=68), 1,000 bootstrap resamples, vs AlphaMissense, ESM1b, phyloP, PROVEAN.

**Calibration.** Rank-align raw predictions to function scores → generalized
additive regression → apply to unmeasured variants → classify at the function-
score cutoffs (−1.360, −0.912).

---

## 7. Deviations register

Maintain this table in the README as the project runs. Every entry is a place
where this reconstruction is knowingly not the paper.

| # | Deviation | Why | Effect |
|---|---|---|---|
| D1 | Training synonymous n=5,031, not 4,395 | Under the exclusion reading that reproduces the paper's missense *and* nonsense counts, none of the 636 missing synonymous variants is accounted for (§1) | ~3% larger training set (22,489 vs 21,853). Synonymous are 5% of each resampled batch, so the effect on headline metrics is small |
| D2 | Coordinates from PDB 8OXQ/7SID, not AlphaFold 3 | AF3 model not deposited; AlphaFold DB has no ATM entry (verified 404) | Unmodelled loops must be masked — exactly the gap problem the paper used AF3 to avoid |
| D3 | Positional encoding added | Not specified in the paper; the encoder otherwise has no ordinal position sense | Likely improves fit; documents a real ambiguity in the methods |
| D4 | dbNSFP v5.1 score versions ≠ the paper's | The paper does not pin a dbNSFP release | Small shifts in the auxiliary features |
| D5 | ClinVar ≥2★ set pinned to release 2026-08-08, not the paper's ~2024 snapshot | ClinVar is versioned weekly and grows | n=70 vs the paper's 68. Size-matching is skipped as uninformative at this margin (§2.3) |
| D6 | Random seeds, PyTorch version, embedding init | Not stated | Run-to-run variance; report seed and mean ± sd over ≥3 seeds |
| D7 | Embeddings combined by summation | The paper says only "integrated with" | Concatenation + projection is an equally valid reading; test both |
| D8 | Windowed attention in CPU smoke mode | Full L=3,056 attention needs GPU memory | Any windowed result is not comparable to the paper; always report the window |

---

## 8. Not in scope

Clinical variant interpretation. This is a study reconstruction. For real ATM
variant calls use the published `Combined_score` / `Classification` columns in
Table S1, or contact the corresponding author (hkim1@yuhs.ac) for the original
code and weights.
