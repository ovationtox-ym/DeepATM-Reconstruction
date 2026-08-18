"""
Load Table S1 (mmc1.xlsx) — "Combined scores (function scores + eDA scores)
for all ATM SNVs" — and reproduce the exact splits described in STAR★Methods
§"Deep learning dataset and feature engineering".

The paper's rule, verbatim:

    "The test dataset comprised all missense variants classified as
     pathogenic, likely pathogenic, benign, or likely benign with a one-star
     rating or higher in ClinVar (n = 116), irrespective of whether they had
     been experimentally evaluated. The training dataset was constructed by
     excluding all evaluated variants that shared amino acid positions with
     the test set. The remainder of the training data consists of evaluated
     missense (n = 16,275), synonymous (n = 4,395) and nonsense variants
     (n = 1,183). Mutations at stop codon positions were excluded."

Table S1's `ClinVar_classification` column is *already* filtered to >=1-star,
so the test set is reconstructible from the supplement alone: selecting
missense rows with a P/LP/B/LB classification yields exactly 116 variants at
103 distinct amino-acid positions, and excluding those positions leaves
exactly 16,275 missense and 1,183 nonsense — both matching the paper.
Synonymous comes out at 5,031 against the paper's 4,395 under the default
missense-only exclusion (see `split_dataset`), so all 636 of the difference is
unexplained; `--position-exclusion coding` removes 174 of them and reaches
4,857, at the cost of nonsense no longer matching. That ~14% difference is
documented, not silently absorbed (EXECUTION_PLAN.md §1, deviation D1).

Outputs (data/processed/):
    train.csv    — measured coding variants, test positions and stop-codon
                   positions excluded. Labels are real function scores.
    test.csv     — the 116 ClinVar missense variants. Evaluation only.
    predict.csv  — the 4,421 rows the paper's own DeepATM predicted. Never
                   used for training; `published_eda_score` is retained so
                   this reconstruction can be scored against the original.
    measured.csv — all 23,092 measured coding rows, pre-exclusion. Used for
                   rank calibration in predict.py.

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

N_RESIDUES = 3056  # full-length ATM; position 3057 is the stop codon

CODING_CONSEQUENCES = ["Missense", "Synonymous", "Nonsense"]

# ClinVar classifications that define the paper's test set. Matched exactly —
# note that "Conflicting classifications of pathogenicity" is deliberately
# absent, and that a substring test for "pathogenic" would wrongly capture it.
CLINVAR_PATHOGENIC = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
CLINVAR_BENIGN = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
}

# The 5 of the paper's 16 auxiliary scores that ship in Table S1.
SUPPLEMENT_SCORE_COLS = [
    "CADD.phred", "boostDM_score", "EVE_scores_ASM", "REVEL", "AlphaMissense",
]

AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}

# Table S1 uses the short form ("M1V", "C11*"); the long HGVS form is
# accepted too so the parser survives a differently-formatted supplement.
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
    """Parse protein changes and normalise the numeric columns.

    Critically, this splits Table S1's `Combined_score` into two disjoint
    columns by provenance. The same column holds an experimentally measured
    function score when `DeepATM_predicted == "No"` and the paper's *own
    DeepATM output* when it is "Yes". Training on the latter would be
    self-supervision on the model being reconstructed, so the two are given
    different names and never merged.
    """
    df = df.copy()
    ref_aa, pos, alt_aa = zip(*df["Protein_change"].map(parse_protein_change))
    df["ref_aa"] = ref_aa
    df["position"] = pd.array(pos, dtype="Int64")
    df["alt_aa"] = alt_aa

    for col in SUPPLEMENT_SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["GnomAD_all_AF"] = pd.to_numeric(df["GnomAD_all_AF"], errors="coerce").fillna(0.0)

    combined = pd.to_numeric(df["Combined_score"], errors="coerce")
    is_predicted = df["DeepATM_predicted"] == "Yes"
    is_measured = df["DeepATM_predicted"] == "No"
    df["function_score"] = combined.where(is_measured)
    df["published_eda_score"] = combined.where(is_predicted)

    return df


def clinvar_label(df: pd.DataFrame) -> pd.Series:
    """Map ClinVar_classification onto 1 (P/LP), 0 (B/LB), or NA."""
    cls = df["ClinVar_classification"]
    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label[cls.isin(CLINVAR_PATHOGENIC)] = 1
    label[cls.isin(CLINVAR_BENIGN)] = 0
    return label


def split_dataset(df: pd.DataFrame, position_exclusion: str = "missense") -> dict[str, pd.DataFrame]:
    """Reproduce the paper's test / train / predict partition.

    `position_exclusion` controls which consequences the "shared amino acid
    position" rule is applied to. The paper says "all evaluated variants",
    but its own counts contradict that:

        applied to all coding : missense 16,275 OK, nonsense 1,148 (paper 1,183)
        applied to missense   : missense 16,275 OK, nonsense 1,183 OK

    Nonsense reproduces exactly only when the rule is *not* applied to it —
    1,192 measured nonsense minus the 9 at the stop codon. So "missense" is
    the reading that matches the published numbers, and is the default.

    "coding" applies the rule to all three consequences. It is the stricter,
    lower-leakage choice — a synonymous variant at a test position still
    trains that position's embedding against a label — at the cost of no
    longer matching the paper. Use it for any claim about generalisation;
    use the default for reproduction.
    """
    if position_exclusion not in ("missense", "coding"):
        raise ValueError(f"position_exclusion must be 'missense' or 'coding', got {position_exclusion!r}")
    df = df.copy()
    df["clinvar_label"] = clinvar_label(df)

    is_coding = df["Variant_consequence"].isin(CODING_CONSEQUENCES)
    is_measured = df["DeepATM_predicted"] == "No"

    # Test set: ALL missense P/LP/B/LB, "irrespective of whether they had been
    # experimentally evaluated" — so this deliberately spans both the measured
    # and the DeepATM-predicted rows.
    test = df[
        (df["Variant_consequence"] == "Missense") & df["clinvar_label"].notna()
    ].reset_index(drop=True)

    measured = df[is_measured & is_coding].reset_index(drop=True)

    # "Mutations at stop codon positions were excluded." Position 3057 is one
    # past the 3,056-residue protein — the terminator itself.
    at_stop_codon = measured["position"] > N_RESIDUES

    # "excluding all evaluated variants that shared amino acid positions with
    # the test set"
    test_positions = set(test["position"].dropna().astype(int))
    shares_test_position = measured["position"].isin(test_positions)
    if position_exclusion == "missense":
        shares_test_position &= measured["Variant_consequence"] == "Missense"

    train = measured[~at_stop_codon & ~shares_test_position].reset_index(drop=True)

    predict = df[
        (df["DeepATM_predicted"] == "Yes") & is_coding & (df["position"] <= N_RESIDUES)
    ].reset_index(drop=True)

    return {
        "train": train,
        "test": test,
        "predict": predict,
        "measured": measured,
    }


# Counts the paper reports, for the provenance check printed by main().
PAPER_COUNTS = {
    ("measured", None): 23092,
    ("test", None): 116,
    ("train", "Missense"): 16275,
    ("train", "Synonymous"): 4395,
    ("train", "Nonsense"): 1183,
    ("predict", None): 4421,
}


def report(splits: dict[str, pd.DataFrame]) -> None:
    """Print each split against the paper's number, flagging any mismatch."""
    def line(label: str, n: int, expected: int | None) -> None:
        if expected is None:
            print(f"  {label:<34}{n:>7}")
        else:
            mark = "OK " if n == expected else "!! "
            print(f"  {mark}{label:<32}{n:>7}   paper {expected:>6,}")

    for name in ("measured", "test", "train", "predict"):
        part = splits[name]
        line(name, len(part), PAPER_COUNTS.get((name, None)))
        if name == "train":
            for cons in CODING_CONSEQUENCES:
                n = int((part["Variant_consequence"] == cons).sum())
                line(f"  {cons.lower()}", n, PAPER_COUNTS.get((name, cons)))

    test = splits["test"]
    print(
        f"\n  test set: {int((test['clinvar_label'] == 1).sum())} P/LP, "
        f"{int((test['clinvar_label'] == 0).sum())} B/LB, "
        f"at {test['position'].nunique()} distinct positions"
    )
    n_scored = int(test["function_score"].notna().sum())
    print(f"  of which {n_scored} were experimentally measured, {len(test) - n_scored} were not")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--position-exclusion", choices=["missense", "coding"], default="missense",
        help="Which consequences the shared-position exclusion applies to. "
             "'missense' reproduces the paper; 'coding' is stricter. See split_dataset().",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        raise FileNotFoundError(
            f"{args.raw} not found. Download the paper's Supplemental Information "
            f"(https://doi.org/10.1016/j.cell.2025.05.046) and place mmc1.xlsx there."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    df = build_dataset(load_table_s1(args.raw))
    splits = split_dataset(df, position_exclusion=args.position_exclusion)

    print(f"Split sizes vs. the paper (position exclusion: {args.position_exclusion}):")
    report(splits)
    print()

    for name, part in splits.items():
        out_path = args.out / f"{name}.csv"
        part.to_csv(out_path, index=False)
        print(f"  wrote {len(part):6d} rows -> {out_path}")


if __name__ == "__main__":
    main()
