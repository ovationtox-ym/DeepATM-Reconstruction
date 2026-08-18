# Running the full reproduction on an AWS GPU

Everything in `outputs/` so far came from a windowed CPU smoke run and is not
comparable to the paper (D8). This is how to produce the run that is.

**Two things to know before planning around AWS:**

1. **The free tier has no GPU instances.** Free tier is `t2/t3.micro` only. What
   a new account gives you is *credits* (currently up to ~$200 — some at signup,
   the rest for completing onboarding activities), and those can be spent on GPU
   instances. "Free" in the credit sense, not the free-tier sense.
2. **New accounts have a G/P instance quota of 0 vCPUs.** You cannot launch a GPU
   instance until you request an increase. This is the long pole — usually hours,
   occasionally a couple of days. **Do step 1 before anything else.**

---

## 1. Request the quota (do this first)

Console → **Service Quotas** → **AWS services** → **Amazon EC2** →
search **"Running On-Demand G and VT instances"** (quota code `L-DB2E81BA`).

Request **4 vCPUs** — one `g4dn.xlarge` or `g5.xlarge`. Asking for a small,
specific number is approved faster than asking for a large one.

If you intend to use spot (cheaper, see below), request the separate
**"All G and VT Spot Instance Requests"** quota as well — the two are not
interchangeable, and having only one is the most common way this step gets
done twice.

Quotas are **per region**. Request it in the region you will actually launch in,
and use that same region throughout. `us-east-1` has the widest capacity.

---

## 2. Pick an instance

The model is small — `embed_dim=64`, 2 encoder layers. Essentially all the cost
is the L=3,056 attention: ~5 GFLOP per sample forward, so the 150-epoch ceiling
across 5 folds × 22,489 rows is ~2.7e17 FLOPs. Early stopping (patience 20)
will realistically cut that by half or more.

| Instance | GPU | On-demand | Est. run | Est. cost |
|---|---|---|---|---|
| **`g4dn.xlarge`** | T4, 16 GB | ~$0.53/hr | 8–12 h worst case, 3–5 h typical | **~$3–6** |
| `g5.xlarge` | A10G, 24 GB | ~$1.01/hr | ~2× faster | ~$4–7 |
| `g6.xlarge` | L4, 24 GB | ~$0.81/hr | similar to g5 | ~$4–6 |

`g4dn.xlarge` is the recommendation: 16 GB is far more memory than this needs,
and it is the cheapest way to get the run done.

Memory is not a constraint. `nn.MultiheadAttention` dispatches to
`scaled_dot_product_attention` on torch ≥ 2.x, so the L×L attention matrix is
never materialised — at B=20, L=3,056 the activations are a few hundred MB.

**Spot** runs ~60–70% cheaper (~$0.16–0.20/hr for `g4dn.xlarge`) and is worth
using *because* training checkpoints every epoch: a reclamation costs one epoch,
and re-running the same command resumes exactly where it stopped. Verified —
an interrupted-and-resumed run reproduces an uninterrupted one bit-for-bit.

Add ~$0.01/hr for a 100 GB `gp3` root volume. Budget the storage separately if
you stop rather than terminate the instance: **a stopped instance still bills
for EBS.**

---

## 3. Launch

Use a **Deep Learning AMI** so the NVIDIA driver and CUDA are already present —
installing them by hand on a bare Ubuntu image is an hour you don't need to
spend. In the launch wizard, search AMIs for:

> Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)

- **Instance type:** `g4dn.xlarge`
- **Key pair:** create one and keep the `.pem`
- **Storage:** 100 GB gp3 (the DLAMI itself is large; ClinVar adds ~190 MB)
- **Security group:** SSH (port 22) from *your IP only*, not `0.0.0.0/0`

Then:

```bash
ssh -i your-key.pem ubuntu@<public-ip>
nvidia-smi          # confirm the GPU is visible before doing anything else
```

---

## 4. Set up

```bash
git clone <your-repo-url> DeepATM-Reconstruction
cd DeepATM-Reconstruction

source activate pytorch          # DLAMI's preinstalled env
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# must print True — if it prints False, stop and fix this first
```

**Upload the supplement from your laptop.** `mmc1.xlsx` is Elsevier/Cell Press
copyright and is deliberately not in the repo (`.gitignore` excludes
`data/raw/*`), so it has to be copied across:

```bash
# from your local machine
scp -i your-key.pem "mmc1.xlsx" ubuntu@<public-ip>:~/DeepATM-Reconstruction/data/raw/
```

---

## 5. Run

Run inside `tmux` so a dropped SSH connection doesn't kill the job:

```bash
tmux new -s deepatm
./scripts/run_full.sh
# detach with Ctrl-B then D; reattach later with: tmux attach -t deepatm
```

The script does the whole pipeline: splits → ClinVar ≥2★ subset → train →
ablation → evaluate → RF baseline → eDA scores → archive. Logs land in
`outputs/logs/`.

**Check the plumbing first** if you like — about ten minutes, and it exercises
every step:

```bash
EPOCHS=2 SMOKE=1 ./scripts/run_full.sh
```

**If the run is interrupted** — spot reclamation, OOM, closed laptop — just run
the same command again. Completed folds are skipped and the in-progress fold
picks up at its last epoch.

Watch progress from a second shell:

```bash
tail -f outputs/logs/train.log
nvidia-smi -l 5      # confirm the GPU is actually busy
```

If GPU utilisation sits low while a CPU core pins at 100%, the DataLoader is the
bottleneck rather than the model — raise `WORKERS` (default 8).

---

## 6. Retrieve results and shut down

The script writes `deepatm-results-<timestamp>.tar.gz` containing `outputs/`
and the five fold checkpoints.

```bash
# from your local machine
scp -i your-key.pem ubuntu@<public-ip>:~/DeepATM-Reconstruction/deepatm-results-*.tar.gz .
```

Then **terminate the instance** — not "stop". A stopped instance keeps billing
for its EBS volume, and there is nothing on it worth keeping once the archive is
down.

```
EC2 console → Instances → select → Instance state → Terminate
```

Set a **billing alarm** before you start (Billing → Budgets, e.g. $20). It is
the cheapest insurance against an instance left running over a weekend.

---

## 7. What to check in the results

The run is only comparable to the paper if `outputs/train_summary.json` shows
`"window_size": null` and `"n_rows": 22489`. Then:

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
