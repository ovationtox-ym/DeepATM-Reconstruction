# Running the full reproduction on a Kaggle GPU

Everything in `outputs/` so far came from a windowed CPU smoke run and is not
comparable to the paper (D8). This is how to produce the run that is, on
Kaggle's free GPU tier.

`notebooks/kaggle_deepatm.ipynb` is the notebook this document describes.
Import it, set the three variables in the first cell, and commit it.

**Why Kaggle rather than a cloud VM:** no quota request, no credit card, no
instance to remember to terminate. 30 GPU-hours per week, 12 hours per session,
and the run is 3–5 hours. The cost of that is a fixed environment and a
save-your-output ritual, both of which are handled below.

**Three things to know before planning around it:**

1. **Internet and GPU both require a phone-verified account.** Settings →
   Phone Verification. Without it the accelerator dropdown is disabled and the
   notebook's "Internet" toggle is greyed out, and the run needs internet for
   `pip`, the PDB structure fetch, and the ClinVar VCF. Do this first.
2. **The 12-hour session cap is per session, not per run.** Training
   checkpoints every epoch and `run_full.sh` resumes, so a run that overruns is
   continued in a second session (§6). It is not lost work.
3. **`/kaggle/working` only survives if the version is committed.** An
   interactive session that you close, or that idles out after 20 minutes, takes
   its files with it. Use **Save Version → Save & Run All (Commit)**, which runs
   the notebook headless to completion and persists the output.

---

## 1. Upload the supplement as a private dataset

`mmc1.xlsx` is Elsevier/Cell Press copyright, which is why `.gitignore` excludes
`data/raw/*` and why it is not in the repo. It has to reach the notebook some
other way, and a Kaggle Dataset is the supported route.

Kaggle → **Datasets** → **New Dataset**:

- Upload `mmc1.xlsx`
- Title: `deepatm-mmc1`
- Visibility: **Private** — not "Public". This matters. A public dataset
  redistributes the supplement.

Note the slug Kaggle assigns (`<your-username>/deepatm-mmc1`); the notebook
needs it. Then in the notebook editor: **Add Input** → **Datasets** → your
dataset. It mounts read-only at `/kaggle/input/deepatm-mmc1/`.

## 2. Create the notebook

Kaggle → **Create** → **New Notebook** → **File** → **Import Notebook** →
upload `notebooks/kaggle_deepatm.ipynb`.

In the right-hand sidebar:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** |
| Persistence | Files only (or off; the commit saves output either way) |

**On the accelerator choice.** The code is single-GPU — only `cuda:0` is used,
and the second T4 idles. Pick T4 anyway: training runs under AMP, and the T4's
fp16 tensor cores (~65 TFLOPS) are roughly 3× the P100's fp16 throughput
(~19 TFLOPS). If T4s are unavailable, P100 works and costs about 2× the wall
clock. Both are pre-Ampere, so PyTorch's SDPA takes the memory-efficient
backend rather than flash; memory is not a constraint either way — at B=20,
L=3,056 the activations are a few hundred MB against 16 GB.

## 3. Set the three variables

Top cell of the notebook:

```python
REPO_URL   = "https://github.com/ovationtox-ym/DeepATM-Reconstruction.git"
MMC1_INPUT = None    # auto-discovered under /kaggle/input; see below
RESUME_FROM = None   # see §6
```

`MMC1_INPUT` is the one people get wrong, which is why the default is now to
not write it down. Kaggle lowercases and slugifies dataset titles, and the
mount layout varies — a dataset lands at either `/kaggle/input/<slug>/` or
`/kaggle/input/datasets/<user>/<slug>/`, and the slug of a dataset titled
`mmc1.xlsx` is the *directory* `mmc1-xlsx`. Left as `None`, the notebook globs
`/kaggle/input/**/mmc1*.xls*`, requires exactly one match, and prints what it
found. Set it explicitly only to override that.

## 4. What the notebook does

1. Asserts `torch.cuda.is_available()` and prints the device. **If this fails,
   stop** — the accelerator was not enabled, and `--full-length` refuses to run
   on CPU (`train.py` requires `--allow-cpu-full-length`, which would take days).
2. `pip install gemmi pygam` — the only two requirements Kaggle's image lacks.
   It deliberately does **not** `pip install -r requirements.txt`, because that
   would reinstall `torch` and replace the CUDA build with whatever pip
   resolves. A verification cell imports every requirement afterwards.
3. Clones the repo to `/kaggle/working/DeepATM-Reconstruction` and copies
   `mmc1.xlsx` into `data/raw/`.
4. Restores checkpoints from a previous version's output, if `RESUME_FROM` is
   set (§6).
5. Runs `WORKERS=2 ./scripts/run_full.sh`. `WORKERS=2` rather than the default
   8 — Kaggle gives 4 vCPUs, and 8 loader workers oversubscribe them and slow
   the run down.
6. Copies `deepatm-results-*.tar.gz`, `outputs/`, and `checkpoints/` to
   `/kaggle/working/` so the commit picks them up, then prints the headline
   table.

Before the real thing, the notebook has an optional smoke cell — set
`RUN_SMOKE = True`, run it, set it back to `False`. About ten minutes, it
exercises every step of the pipeline, and its results are *not* comparable to
the paper. Worth running once to confirm the dataset path and internet access
before spending a 12-hour commit on it.

**The smoke run deletes its own checkpoints when it finishes, and that cleanup
is not optional.** `train.py` fingerprints every resume file with the epoch
count, window size and split sizes, and refuses to resume across a mismatch —
a hard error rather than a silent fresh start, since resuming into a checkpoint
trained under other settings would yield a model matching neither
configuration. A surviving smoke resume file therefore *aborts* the full run at
step 3 instead of being ignored. If you ever run the smoke pass by hand rather
than through the cell, clear them yourself:

```bash
rm -f checkpoints/resume_fold*.pt checkpoints/deepatm_fold*.pt
```

## 5. Run it

**Save Version** → **Save & Run All (Commit)** → Save.

The session then runs headless; you can close the tab. Progress is visible from
the notebook's **Versions** list (click the running version → its log). Expect
3–5 hours on T4.

Do not just press "Run All" in the interactive editor and walk away: an
interactive session idles out after 20 minutes of no browser interaction, and
takes `/kaggle/working` with it.

## 6. If it doesn't finish in 12 hours

The version is saved with whatever `/kaggle/working` held when the cap hit —
which includes `checkpoints/`, because the notebook copies them out after each
`run_full.sh` step. To continue:

1. In the notebook editor: **Add Input** → **Notebook Output** → select your
   own previous version.
2. Set `RESUME_FROM = "/kaggle/input/<notebook-slug>/checkpoints"` in the first
   cell.
3. Commit again.

Completed folds are skipped and the in-progress fold restarts from its last
epoch. A resumed run reproduces an uninterrupted one bit-for-bit — the resume
file carries a fingerprint of the run-defining flags and `train.py` refuses to
resume across a settings change rather than silently mixing two runs.

Watch the weekly quota (**Settings → Accelerator quota**, resets Saturday
00:00 UTC). A completed run plus one resumed attempt fits comfortably in 30
hours; three failed 12-hour attempts do not.

## 7. Retrieve the results

Version page → **Output** tab → download `deepatm-results-<timestamp>.tar.gz`
(it holds `outputs/` and the five fold checkpoints), or download individual
files from `outputs/`.

Then, locally:

```bash
tar xzf deepatm-results-*.tar.gz
```

Nothing needs shutting down — Kaggle sessions end themselves, and there is no
storage that keeps billing.

## 8. What to check in the results

The run is only comparable to the paper if `outputs/train_summary.json` shows
`"window_size": null` and `"n_rows": 21715`. Then:

| Quantity | Paper | Where |
|---|---|---|
| 5-fold CV Pearson r | ≈ 0.61 | `metrics.json → cross_validation.median_pearson` |
| auROC, ClinVar ≥1★ (n=116) | 0.95 | `metrics.json → clinvar_1star.deepatm.auroc` |
| auROC, ClinVar ≥2★ (n=68; ours n=70) | — | `metrics.json → clinvar_2star.auroc` |
| eDA ↔ published eDA | 0.70 | `predict_summary.json → vs_published_eda.pearson` |
| Structural ablation | p = 0.032 | `ablation_comparison.json → paired_bootstrap.p_value` |
| RF baseline | ≈ 0.55 | `baseline_rf.json` |

The ablation matters as much as the headline r. A model that reaches r ≈ 0.61
with *no* structure gap has probably learned something other than what DeepATM
learned.

## 9. Known Kaggle-specific failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | Accelerator set to None | Sidebar → Accelerator → GPU T4 x2, then restart the session |
| `pip` or the ClinVar download hangs, then fails | Internet toggle off | Sidebar → Internet → On (needs phone verification) |
| `FileNotFoundError` on `mmc1.xlsx` | Dataset not attached, or `MMC1_INPUT` overridden with a guessed path | Leave `MMC1_INPUT = None` and let §4 discover it; check the dataset is attached under Add Input |
| Cell 2 sits for many minutes with no output | Session queuing for a free GPU — no cell is executing yet | Nothing to fix; watch the session status indicator. Switching to P100 often gets a slot sooner |
| Session dies around 20 min with no error | Interactive idle timeout | Use Save & Run All (Commit), not Run All |
| `outputs/` empty after the session | Version was never committed | Same as above |
| Run is slow and `nvidia-smi` shows low GPU use | DataLoader oversubscribing 4 vCPUs | Keep `WORKERS=2`; do not raise it |
| Step 3 aborts with "written under different settings and cannot be resumed" | Smoke-run resume files survived into the full run | `rm -f checkpoints/resume_fold*.pt checkpoints/deepatm_fold*.pt`, then re-run. The smoke cell does this automatically |
| `pygam` install fails on the resolver | pygam pins older scipy in some images | `pip install --no-deps pygam` — it only needs numpy/scipy at runtime, both present |

---

`docs/aws-gpu-run.md` documents the same run on an AWS GPU instance. The two are
interchangeable; the numbers in §8 do not depend on which was used.
