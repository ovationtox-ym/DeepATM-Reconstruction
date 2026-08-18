#!/usr/bin/env bash
#
# The full-scale reproduction, start to finish, unattended.
#
# This is the run whose numbers are comparable to the paper: all 22,489
# training rows, the full 3,056-residue sequence (no window), 150 epochs with
# early stopping, 5 folds. Every earlier result in outputs/ came from a
# windowed CPU smoke run and is not comparable (D8).
#
# Designed to be safe to re-run. Training checkpoints every epoch, so an
# interrupted run — a spot reclamation, a dropped SSH session, an OOM — is
# continued by invoking this script again with the same arguments. Completed
# folds are skipped, and the in-progress fold restarts from its last epoch.
# A resumed run reproduces an uninterrupted one bit-for-bit.
#
# Usage:
#   ./scripts/run_full.sh                 # the paper's settings
#   EPOCHS=5 SMOKE=1 ./scripts/run_full.sh   # ~10 min sanity pass, windowed
#
# Environment:
#   EPOCHS        max epochs per fold (default 150)
#   FOLDS         cross-validation folds (default 5)
#   SEED          random seed (default 0)
#   WORKERS       DataLoader workers (default 8)
#   SMOKE         if set, run windowed on a subsample — for checking plumbing
#   SKIP_ABLATION if set, skip the no-coordinates arm
#
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-150}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"

if [[ -n "${SMOKE:-}" ]]; then
  TRAIN_MODE=(--window 65 --max-rows 800)
  echo "SMOKE MODE — windowed and subsampled. Results are NOT comparable to the paper."
else
  TRAIN_MODE=(--full-length)
fi

COMMON=(--epochs "$EPOCHS" --folds "$FOLDS" --seed "$SEED" \
        --num-workers "$WORKERS" --resume "${TRAIN_MODE[@]}")

LOG_DIR=outputs/logs
mkdir -p "$LOG_DIR"

step() { echo; echo "=== $* ==="; date -u +"    started %Y-%m-%dT%H:%M:%SZ"; }

step "0/8  environment"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}, {p.total_memory / 1e9:.0f} GB, capability {p.major}.{p.minor}")
else:
    print("  WARNING: no GPU. --full-length will refuse to run.")
PY

step "1/8  splits from Table S1"
python -m src.data_prep 2>&1 | tee "$LOG_DIR/data_prep.log"

step "2/8  ClinVar >=2-star test subset"
# ~190 MB on the first run, cached after. Non-fatal: the >=1-star test set is
# the headline number and does not depend on this.
python -m src.clinvar 2>&1 | tee "$LOG_DIR/clinvar.log" || \
  echo "WARNING: >=2-star subset unavailable; evaluation will report >=1-star only"

step "3/8  train (with structural coordinates)"
python -m src.train "${COMMON[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"

if [[ -z "${SKIP_ABLATION:-}" ]]; then
  step "4/8  train ablation (no coordinates)"
  # Same seed and fold count, so the splits are identical to the run above and
  # the comparison below is genuinely paired.
  python -m src.train "${COMMON[@]}" --no-coordinates --tag nocoord 2>&1 \
    | tee -a "$LOG_DIR/train_nocoord.log"
fi

step "5/8  evaluate"
python -m src.evaluate --seed "$SEED" 2>&1 | tee "$LOG_DIR/evaluate.log"

if [[ -z "${SKIP_ABLATION:-}" ]]; then
  step "6/8  structural ablation (M6)"
  python -m src.evaluate --tag nocoord --skip-test-set 2>&1 \
    | tee "$LOG_DIR/evaluate_nocoord.log"
  python -m src.compare_runs \
    outputs/oof_predictions.csv outputs/oof_predictions_nocoord.csv \
    --label-a coords --label-b nocoord --seed "$SEED" \
    --out outputs/ablation_comparison.json 2>&1 | tee "$LOG_DIR/ablation.log"
fi

step "7/8  random-forest baseline (M6)"
python -m src.baseline_rf --folds "$FOLDS" --seed "$SEED" 2>&1 | tee "$LOG_DIR/baseline_rf.log"

step "8/8  eDA scores for the 4,421 unevaluated variants (M7)"
python -m src.predict 2>&1 | tee "$LOG_DIR/predict.log"

echo
echo "=== done ==="
ARCHIVE="deepatm-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar czf "$ARCHIVE" outputs checkpoints/deepatm_fold*.pt
echo "results archived to $ARCHIVE"
echo
echo "Headline numbers vs the paper:"
python - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path("outputs/metrics.json").read_text())
cv, targets = m["cross_validation"], m["paper_targets"]
print(f"  CV Pearson r      {cv['median_pearson']:.3f}   paper {targets['cv_pearson']:.2f}")
one = m.get("clinvar_1star", {}).get("deepatm", {})
if one.get("auroc"):
    print(f"  auROC >=1 star    {one['auroc']:.3f}   paper {targets['test_auroc_1star']:.2f}")
two = m.get("clinvar_2star") or {}
if two.get("auroc"):
    print(f"  auROC >=2 star    {two['auroc']:.3f}   (n={two['n']}, paper n={two.get('paper_n')})")
p = pathlib.Path("outputs/predict_summary.json")
if p.exists():
    eda = json.loads(p.read_text()).get("vs_published_eda", {})
    if eda:
        print(f"  vs published eDA  {eda['pearson']:.3f}   paper {targets['eda_correlation']:.2f}")
w = m.get("ensemble", {}).get("window_size")
print(f"  window            {w if w is not None else 'full length (comparable to the paper)'}")
PY
