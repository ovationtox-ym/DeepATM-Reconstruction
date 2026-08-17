"""Tests for the ClinVar >=2-star join.

None of these touch the 193 MB release file — the parsing and filtering rules
are exercised on synthetic records, which is where the mistakes actually live.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest

from src.clinvar import (
    build_2star_subset,
    clnsig_label,
    join_stars,
    parse_atm_records,
    parse_info,
    review_stars,
)


class TestReviewStars:
    @pytest.mark.parametrize("status,expected", [
        ("practice_guideline", 4),
        ("reviewed_by_expert_panel", 3),
        ("criteria_provided,_multiple_submitters,_no_conflicts", 2),
        ("criteria_provided,_single_submitter", 1),
        ("criteria_provided,_conflicting_classifications", 1),
        ("no_assertion_criteria_provided", 0),
    ])
    def test_known_statuses(self, status, expected):
        assert review_stars(status) == expected

    def test_conflicting_is_one_star_not_two(self):
        """A conflicting record has multiple submitters but is 1 star, and it
        is the one most likely to be misfiled as 2 by a substring rule."""
        assert review_stars("criteria_provided,_conflicting_classifications") == 1

    @pytest.mark.parametrize("value", [None, float("nan"), "", "something_new"])
    def test_missing_or_unknown_is_zero(self, value):
        """Unrecognised statuses must fail closed — 0 keeps them out of a
        >=2-star filter rather than smuggling them in."""
        assert review_stars(value) == 0


class TestClnsigLabel:
    @pytest.mark.parametrize("clnsig,expected", [
        ("Pathogenic", 1),
        ("Likely_pathogenic", 1),
        ("Pathogenic/Likely_pathogenic", 1),
        ("Benign", 0),
        ("Likely_benign", 0),
        ("Benign/Likely_benign", 0),
    ])
    def test_labelled(self, clnsig, expected):
        assert clnsig_label(clnsig) == expected

    def test_conflicting_is_unlabelled(self):
        """Contains the substring 'pathogenicity'; must not score as pathogenic."""
        assert clnsig_label("Conflicting_classifications_of_pathogenicity") is None

    @pytest.mark.parametrize("value", ["Uncertain_significance", None, float("nan"), ""])
    def test_unlabelled(self, value):
        assert clnsig_label(value) is None


def test_parse_info_handles_flags_and_pairs():
    info = parse_info("ALLELEID=12;CLNSIG=Pathogenic;RS=1;SOMATIC")
    assert info["CLNSIG"] == "Pathogenic"
    assert info["SOMATIC"] == ""


def _vcf(tmp_path, records):
    path = tmp_path / "clinvar.vcf.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.1\n##fileDate=2026-08-08\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, vid, ref, alt, info in records:
            fh.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t{info}\n")
    return path


SNV = "CLNVC=single_nucleotide_variant"


def test_parse_atm_records_filters_locus_and_variant_type(tmp_path):
    path = _vcf(tmp_path, [
        ("11", 108236000, "1", "A", "G", f"{SNV};CLNSIG=Pathogenic;"
         "CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=ATM:472"),
        ("11", 500, "2", "A", "G", f"{SNV};CLNSIG=Benign"),          # off-locus
        ("17", 108236000, "3", "A", "G", f"{SNV};CLNSIG=Benign"),    # wrong chrom
        ("11", 108236100, "4", "AT", "A", "CLNVC=Deletion;CLNSIG=Benign"),  # not an SNV
    ])
    records, file_date = parse_atm_records(path)

    assert file_date == "2026-08-08"
    assert list(records["clinvar_variation_id"]) == ["1"]
    assert records.loc[0, "stars"] == 3
    assert records.loc[0, "clinvar_label_current"] == 1


def test_parse_atm_records_reads_non_utf8_safe(tmp_path):
    """The real file carries non-ASCII submitter names; text mode must not
    fall back to the locale encoding, which is how this first broke."""
    path = _vcf(tmp_path, [
        ("11", 108236000, "1", "A", "G",
         f"{SNV};CLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;"
         "CLNDN=Ataxia–telangiectasia"),
    ])
    records, _ = parse_atm_records(path)
    assert len(records) == 1


def test_parse_atm_records_rejects_a_file_with_no_atm(tmp_path):
    path = _vcf(tmp_path, [("17", 43000000, "1", "A", "G", f"{SNV};CLNSIG=Benign")])
    with pytest.raises(ValueError, match="no ATM-locus SNVs"):
        parse_atm_records(path)


@pytest.fixture
def test_frame():
    """Four rows standing in for the 116-variant >=1-star test set."""
    return pd.DataFrame({
        "Chrom": ["11"] * 4,
        "hg38_pos": [108236001, 108236002, 108236003, 108236004],
        "Ref": ["A", "C", "G", "T"],
        "Alt": ["G", "T", "A", "C"],
        "clinvar_label": [1, 0, 1, 1],
    })


@pytest.fixture
def record_frame():
    return pd.DataFrame({
        "Chrom": ["11"] * 3,
        "hg38_pos": [108236001, 108236002, 108236003],
        "Ref": ["A", "C", "G"],
        "Alt": ["G", "T", "A"],
        "clinvar_variation_id": ["1", "2", "3"],
        "clnsig": ["Pathogenic", "Benign", "Uncertain_significance"],
        "clnrevstat": [
            "criteria_provided,_multiple_submitters,_no_conflicts",
            "criteria_provided,_single_submitter",
            "reviewed_by_expert_panel",
        ],
        "stars": [2, 1, 3],
        "clinvar_label_current": pd.array([1, 0, None], dtype="Int64"),
    })


def test_join_leaves_unmatched_rows_at_zero_stars(test_frame, record_frame):
    merged = join_stars(test_frame, record_frame)
    assert len(merged) == len(test_frame)
    assert merged.loc[3, "stars"] == 0  # 108236004 is absent from ClinVar


def test_subset_keeps_only_two_star_still_labelled(test_frame, record_frame):
    subset = build_2star_subset(join_stars(test_frame, record_frame))
    assert list(subset["hg38_pos"]) == [108236001]


def test_subset_excludes_reclassified_high_star_variants(test_frame, record_frame):
    """108236003 is 3-star but now Uncertain_significance — high confidence in
    a classification that is no longer P/LP or B/LB gives nothing to score."""
    subset = build_2star_subset(join_stars(test_frame, record_frame))
    assert 108236003 not in set(subset["hg38_pos"])


def test_join_does_not_fan_out_on_shared_protein_change(test_frame, record_frame):
    """The key is the SNV, not the amino-acid change; a one_to_one join is
    asserted inside join_stars, so a duplicated key would raise here."""
    merged = join_stars(test_frame, record_frame)
    assert merged["hg38_pos"].is_unique
