"""
Build the ClinVar >=2-star test subset, the one split that Table S1 cannot
supply on its own.

STAR★Methods reports auROC on two test sets: all missense variants classified
P/LP/B/LB at >=1 star (n = 116), and the >=2-star subset of those (n = 68).
Table S1's `ClinVar_classification` column is already >=1-star filtered by the
authors, so the 116 are reconstructible from the supplement alone
(`data_prep.split_dataset`). It carries no review-status column, though, so
the 68 require going back to ClinVar itself.

Source
------
The GRCh38 release VCF, not `variant_summary.txt.gz`:

    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

Both files carry review status. The VCF is preferred here because its
coordinates are already GRCh38 VCF-normalised, which makes the join to Table
S1's `hg38_pos`/`Ref`/`Alt` exact rather than assembly-filtered-then-hoped-for;
it is roughly a third of the size; and its `##fileDate` header pins the release
without a separate lookup. Only the ~150 kb ATM locus is parsed.

Pinning
-------
ClinVar is re-released weekly and the >=2-star set grows monotonically as
submissions accumulate. An unpinned test set is not a test set, so the release
date, source URL and file digest are written to
`data/processed/clinvar_release.json` next to the subset itself, and
`evaluate.py` echoes them into `outputs/metrics.json`. Expect more than the
paper's n = 68: their snapshot is ~2024 (D5).

Usage:
    python -m src.clinvar                  # download (cached), build the subset
    python -m src.clinvar --vcf path.gz    # use an already-downloaded release
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .data_prep import CLINVAR_BENIGN, CLINVAR_PATHOGENIC

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CLINVAR_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
VCF_PATH = RAW_DIR / "clinvar_GRCh38.vcf.gz"

# ATM, GRCh38: chr11:108,222,500-108,369,100 (EXECUTION_PLAN.md 2.1). Padded
# slightly; the join to Table S1 is what actually selects the variants.
ATM_CHROM = "11"
ATM_START = 108_220_000
ATM_END = 108_372_000

# ClinVar review status -> star rating. The VCF writes CLNREVSTAT with spaces
# replaced by underscores. Anything not listed here is treated as 0 stars,
# which is the safe direction for a >=2-star filter.
REVIEW_STATUS_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,  # pre-2024 spelling
    "no_assertion_criteria_provided": 0,
    "no_assertion_provided": 0,
    "no_classification_provided": 0,
    "no_classification_for_the_single_variant": 0,
    "no_interpretation_for_the_single_variant": 0,  # pre-2024 spelling
}

# CLNSIG uses underscores where data_prep's Table S1 labels use spaces and
# slashes. Mapped back so both sources are judged by one definition.
_CLNSIG_TO_LABEL = {v.replace(" ", "_"): 1 for v in CLINVAR_PATHOGENIC}
_CLNSIG_TO_LABEL.update({v.replace(" ", "_"): 0 for v in CLINVAR_BENIGN})


def review_stars(clnrevstat: str | None) -> int:
    """Star rating for a CLNREVSTAT value; 0 for anything unrecognised."""
    if not isinstance(clnrevstat, str):  # absent INFO field -> NaN, not None
        return 0
    return REVIEW_STATUS_STARS.get(clnrevstat.strip(), 0)


def clnsig_label(clnsig: str | None) -> int | None:
    """1 for P/LP, 0 for B/LB, None otherwise.

    Matched against an explicit set, so "Conflicting_classifications_of_
    pathogenicity" is excluded rather than caught by a substring test — the
    same trap `data_prep.clinvar_label` avoids on the Table S1 side.
    """
    if not isinstance(clnsig, str):
        return None
    return _CLNSIG_TO_LABEL.get(clnsig.strip())


def _urllib_download(url: str, tmp: Path, context: ssl.SSLContext | None) -> None:
    with urllib.request.urlopen(url, context=context, timeout=60) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            seen += len(chunk)
            if total and seen % (1 << 25) < (1 << 20):
                print(f"    {seen / 1e6:6.0f} / {total / 1e6:.0f} MB", flush=True)


def _curl_download(url: str, tmp: Path) -> None:
    """Fetch with the system curl, which uses the platform TLS stack.

    On Windows that is Schannel, which reads the same trust store OpenSSL
    chokes on but tolerates the malformed roots that TLS-inspecting security
    software installs. Verification is left on — no `-k`.
    """
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl not found")
    subprocess.run(
        [curl, "-fSL", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url],
        check=True,
    )


def _download_attempts():
    """TLS strategies, in order of preference. Each verifies the certificate.

    The system trust store is tried first. Some machines carry a root whose
    Basic Constraints extension is not marked critical, which OpenSSL 3
    rejects outright, and a TLS-inspecting proxy in front of the connection
    defeats certifi's bundle too — hence the platform-TLS fallback via curl.
    """
    yield "system trust store", lambda url, tmp: _urllib_download(url, tmp, None)

    try:
        import certifi
    except ImportError:
        pass
    else:
        yield "certifi bundle", lambda url, tmp: _urllib_download(
            url, tmp, ssl.create_default_context(cafile=certifi.where())
        )

    yield "system curl", _curl_download


def download_vcf(url: str = CLINVAR_VCF_URL, dest: Path = VCF_PATH, force: bool = False) -> Path:
    """Fetch the release VCF, streaming to disk. Cached unless `force`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"using cached {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print(f"downloading {url}\n  -> {dest}  (~190 MB, one time)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for name, attempt in _download_attempts():
        try:
            attempt(url, tmp)
            break
        except (urllib.error.URLError, ssl.SSLError, OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{name}: {exc}")
            tmp.unlink(missing_ok=True)
            print(f"  {name} failed, trying the next transport")
    else:
        raise RuntimeError(
            "could not download the ClinVar VCF.\n  "
            + "\n  ".join(errors)
            + f"\nDownload {url} manually and pass --vcf <path>."
        )

    tmp.replace(dest)
    print(f"  done: {dest.stat().st_size / 1e6:.0f} MB")
    return dest


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_info(info: str) -> dict[str, str]:
    out = {}
    for field in info.split(";"):
        key, sep, value = field.partition("=")
        out[key] = value if sep else ""
    return out


def parse_atm_records(vcf_path: Path = VCF_PATH) -> tuple[pd.DataFrame, str]:
    """Stream the VCF, returning ATM-locus SNV records and the release date.

    Only `chr11:ATM_START-ATM_END` is retained, so the whole file is read once
    but almost nothing is kept.
    """
    rows = []
    file_date = "unknown"
    # Explicit UTF-8: gzip's text mode otherwise follows the locale encoding,
    # which fails outright on a non-UTF-8 default (cp949, cp1252, ...). The
    # VCF carries non-ASCII in submitter names and condition strings.
    with gzip.open(vcf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("##fileDate="):
                    file_date = line.strip().split("=", 1)[1]
                continue
            chrom, pos, _id, ref, alt, _qual, _filter, info = line.rstrip("\n").split("\t")[:8]
            if chrom != ATM_CHROM:
                continue
            position = int(pos)
            if not (ATM_START <= position <= ATM_END):
                continue
            fields = parse_info(info)
            if fields.get("CLNVC") != "single_nucleotide_variant":
                continue
            rows.append({
                "Chrom": chrom,
                "hg38_pos": position,
                "Ref": ref,
                "Alt": alt,
                "clinvar_variation_id": _id,
                "clnsig": fields.get("CLNSIG"),
                "clnrevstat": fields.get("CLNREVSTAT"),
                "geneinfo": fields.get("GENEINFO", ""),
                "molecular_consequence": fields.get("MC", ""),
            })

    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError(
            f"no ATM-locus SNVs found in {vcf_path}. Is this a GRCh38 ClinVar VCF?"
        )
    records["stars"] = records["clnrevstat"].map(review_stars)
    records["clinvar_label_current"] = records["clnsig"].map(clnsig_label).astype("Int64")
    return records, file_date


JOIN_KEY = ["Chrom", "hg38_pos", "Ref", "Alt"]


def join_stars(test_df: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Left-join current ClinVar status onto the 116-variant test set.

    The join is on (Chrom, hg38_pos, Ref, Alt) — the SNV identity. Protein
    change is not a key: several distinct SNVs share one amino-acid change.
    Unmatched rows get 0 stars, the safe direction for a >=2-star filter.
    """
    left = test_df.copy()
    left["Chrom"] = left["Chrom"].astype(str)
    left["hg38_pos"] = left["hg38_pos"].astype(int)

    right = records[JOIN_KEY + ["clinvar_variation_id", "clnsig", "clnrevstat",
                               "stars", "clinvar_label_current"]].drop_duplicates(subset=JOIN_KEY)

    merged = left.merge(right, on=JOIN_KEY, how="left", validate="one_to_one")
    merged["stars"] = merged["stars"].fillna(0).astype(int)
    return merged


def build_2star_subset(merged: pd.DataFrame, min_stars: int = 2) -> pd.DataFrame:
    """Filter a `join_stars` frame to the high-confidence, still-labelled rows.

    A variant is kept when it is >= `min_stars` AND its *current* ClinVar
    classification is still P/LP or B/LB. Both conditions matter: a variant
    can reach 2 stars and, in a later release than the paper's, have been
    reclassified to VUS, in which case it no longer has a label to score.
    Reclassifications are counted and reported rather than silently dropped.
    """
    return merged[
        (merged["stars"] >= min_stars) & merged["clinvar_label_current"].notna()
    ].reset_index(drop=True)


def summarise(test_df: pd.DataFrame, merged_all: pd.DataFrame, subset: pd.DataFrame) -> dict:
    unmatched = int(merged_all["clnsig"].isna().sum())
    relabelled = int(
        (
            merged_all["clinvar_label_current"].notna()
            & (merged_all["clinvar_label_current"] != merged_all["clinvar_label"])
        ).sum()
    )
    dropped = int(
        ((merged_all["stars"] >= 2) & merged_all["clinvar_label_current"].isna()).sum()
    )
    return {
        "n_test_1star": int(len(test_df)),
        "n_matched_in_clinvar": int(len(merged_all) - unmatched),
        "n_unmatched": unmatched,
        "n_2star": int(len(subset)),
        "n_2star_pathogenic": int((subset["clinvar_label_current"] == 1).sum()),
        "n_2star_benign": int((subset["clinvar_label_current"] == 0).sum()),
        "n_2star_no_longer_labelled": dropped,
        "n_reclassified_vs_table_s1": relabelled,
        "paper_n_2star": 68,
        "star_histogram": {
            str(k): int(v) for k, v in merged_all["stars"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path, default=VCF_PATH,
                        help="ClinVar GRCh38 VCF; downloaded and cached if absent")
    parser.add_argument("--url", default=CLINVAR_VCF_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--test-csv", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--min-stars", type=int, default=2)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "test_2star.csv")
    parser.add_argument("--release-json", type=Path,
                        default=PROCESSED_DIR / "clinvar_release.json")
    args = parser.parse_args()

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"{args.test_csv} not found — run `python -m src.data_prep` first."
        )

    vcf_path = download_vcf(args.url, args.vcf, force=args.force_download)
    records, file_date = parse_atm_records(vcf_path)
    print(f"ClinVar release {file_date}: {len(records):,} SNVs at the ATM locus")

    test_df = pd.read_csv(args.test_csv)
    test_df = test_df[test_df["clinvar_label"].notna()].reset_index(drop=True)

    merged_all = join_stars(test_df, records)
    subset = build_2star_subset(merged_all, min_stars=args.min_stars)
    stats = summarise(test_df, merged_all, subset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.out, index=False)

    release = {
        "source_url": args.url,
        "file_date": file_date,
        "sha256": file_digest(vcf_path),
        "min_stars": args.min_stars,
        **stats,
    }
    with open(args.release_json, "w") as fh:
        json.dump(release, fh, indent=2)

    print(f"\n>={args.min_stars}-star subset of the {stats['n_test_1star']}-variant test set")
    print(f"  matched in ClinVar        {stats['n_matched_in_clinvar']:>4}"
          f"   ({stats['n_unmatched']} not found)")
    print(f"  >= {args.min_stars} stars and still labelled {stats['n_2star']:>4}"
          f"   paper n=68")
    print(f"    P/LP {stats['n_2star_pathogenic']}, B/LB {stats['n_2star_benign']}")
    if stats["n_2star_no_longer_labelled"]:
        print(f"  >= {args.min_stars} stars but reclassified to VUS/conflicting: "
              f"{stats['n_2star_no_longer_labelled']} (excluded — no label to score)")
    if stats["n_reclassified_vs_table_s1"]:
        print(f"  label differs from Table S1 in {stats['n_reclassified_vs_table_s1']} "
              "variant(s) — ClinVar has moved since the paper's snapshot")
    print(f"  star histogram: {stats['star_histogram']}")
    print(f"\nwrote {args.out}\nwrote {args.release_json}  (release {file_date})")


if __name__ == "__main__":
    main()
