"""
Load Table S1 (mmc1.xlsx) — "Combined scores (function scores + eDA scores)
for all ATM SNVs" — and split it into the sets DeepATM's own paper describes:

  * measured  : DeepATM_predicted == "No"  -> experimentally derived function
                score (from the prime-editing / olaparib screen). This is
                the label DeepATM was trained to regress.
  * predicted : DeepATM_predicted == "Yes" -> the paper's own DeepATM output
                (eDA score) for the 4,421 SNVs that couldn't be edited.
                We keep this aside as a reference set: this reconstruction's
                predictions can be compared against it, but it is never used
                for training.
  * unusable  : DeepATM_predicted == "NA"  -> rows with no protein change
                (e.g. intronic/splice variants that don't map to a single
                residue) or otherwise out of scope for this residue-level
                regression model. Kept for completeness but unused here.

Row-level parsing follows the STAR★Methods "Deep learning dataset and
feature engineering" section: the training set excludes stop-codon rows,
and (in the original paper) any variant sharing an amino-acid position with
the ClinVar-derived test set. We reproduce the ClinVar-based held-out split
in `features.py`/`train.py`; this module only does the raw load + basic
train/val/predict partition.

Usage:
    python -m src.data_prep
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/mmc1.xlsx")
OUT_DIR = Path("data/processed")

AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}

# Matches protein HGVS-ish "p.Met1Lys" or short "M1K" forms found in the
# supplement's `Protein_change` column.
_PROT_RE_LONG = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*)")
_PROT_RE_SHORT = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")


def parse_protein_change(value: str | None) -> tuple[str | None, int | None, str | None]:
    """Return (ref_aa, position, alt_aa) as single-letter codes, or (None, None, None)."""
    if not value or not isinstance(value, str):
        return None, None, None
    m = _PROT_RE_LONG.search(value)
    if m:
        ref3, pos, alt3 = m.groups()
        ref = AA_3TO1.get(ref3, None)
        alt = "*" if alt3 == "*" else AA_3TO1.get(alt3, None)
        return ref, int(pos), alt
    m = _PROT_RE_SHORT.match(value.strip())
    if m:
        ref, pos, alt = m.groups()
        return ref.upper(), int(pos), alt.upper()
    return None, None, None


def load_table_s1(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Table S1", header=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ref_aa, pos, alt_aa = zip(*df["Protein_change"].map(parse_protein_change))
    df = df.copy()
    df["ref_aa"] = ref_aa
    df["position"] = pos
    df["alt_aa"] = alt_aa

    # numeric feature columns present in the supplement (5 of the paper's 16)
    for col in ["CADD.phred", "boostDM_score", "EVE_scores_ASM", "REVEL", "AlphaMissense"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["GnomAD_all_AF"] = pd.to_numeric(df["GnomAD_all_AF"], errors="coerce").fillna(0.0)

    return df


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    measured = df[df["DeepATM_predicted"] == "No"].reset_index(drop=True)
    predicted = df[df["DeepATM_predicted"] == "Yes"].reset_index(drop=True)
    unusable = df[~df["DeepATM_predicted"].isin(["No", "Yes"])].reset_index(drop=True)

    # DeepATM (per STAR Methods) is trained/evaluated on missense, synonymous,
    # and nonsense variants only (stop-codon positions in the *target* are
    # excluded, but nonsense inputs used at low frequency for calibration).
    measured = measured[measured["Variant_consequence"].isin(["Missense", "Synonymous", "Nonsense"])]

    # STAR Methods: "Mutations at stop codon positions were excluded."
    # A handful of rows in the supplement encode readthrough-past-the-stop
    # variants at position 3057 (one past the 3,056-residue protein); those
    # fall outside the reference sequence entirely and are dropped here too.
    from .features import N_RESIDUES
    measured = measured[measured["position"] <= N_RESIDUES]
    predicted = predicted[predicted["position"] <= N_RESIDUES]

    return {"measured": measured, "predicted": predicted, "unusable": unusable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_table_s1(args.raw)
    df = build_dataset(df)
    splits = split_dataset(df)

    for name, part in splits.items():
        out_path = args.out / f"{name}.csv"
        part.to_csv(out_path, index=False)
        print(f"{name:>10}: {len(part):6d} rows -> {out_path}")


if __name__ == "__main__":
    main()
