"""
Data Alignment Pipeline
=======================
Aligns MC3 mutation data with TCGA pathology reports for downstream
prediction of targetable prognostic genomic alterations.

Memory-safe: reads the 3.5 GB MAF file in chunks, never loads it fully.
Saves checkpoints so the MAF never needs to be re-read.
"""

import os
import re
import time
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────
MAF_FILE = "mc3.v0.2.8.PUBLIC.maf"
REPORTS_FILE = "TCGA_Reports.csv"
OUTPUT_DIR = "./output"
CHUNK_SIZE = 50_000  # rows per chunk (kept small for memory safety)

MAF_COLS = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"]
PATIENT_ID_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
PATIENT_ID_EXTRACT_RE = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)")


def validate_patient_ids(ids, source_name):
    """Check every Patient_ID matches the expected TCGA-XX-YYYY format."""
    invalid = [pid for pid in ids if not PATIENT_ID_RE.match(str(pid))]
    if invalid:
        print(f"  [WARN] {len(invalid)} invalid Patient_IDs in {source_name}. "
              f"Examples: {invalid[:5]}")
    else:
        print(f"  [OK]   All Patient_IDs in {source_name} pass format check.")
    return invalid


def main():
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ====================================================================
    # STEP 1 — Chunked loading of the MAF mutation file
    # ====================================================================
    print("=" * 65)
    print("STEP 1: Loading mutation data (chunked)")
    print("=" * 65)

    mutation_rows = []          # accumulate lightweight rows
    total_mutations = 0
    chunks_read = 0

    reader = pd.read_csv(
        MAF_FILE,
        sep="\t",
        comment="#",
        usecols=MAF_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        dtype={"Hugo_Symbol": str,
               "Tumor_Sample_Barcode": str,
               "Variant_Classification": str},
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["Tumor_Sample_Barcode"])
        chunk["Patient_ID"] = chunk["Tumor_Sample_Barcode"].str[:12]
        mutation_rows.append(
            chunk[["Hugo_Symbol", "Variant_Classification", "Patient_ID"]]
        )
        total_mutations += len(chunk)
        chunks_read += 1
        if chunks_read % 20 == 0:
            print(f"  ... chunk {chunks_read:>4}  |  "
                  f"rows so far: {total_mutations:>10,}  |  "
                  f"mem chunks held: {len(mutation_rows)}")

    mutations_df = pd.concat(mutation_rows, ignore_index=True)
    del mutation_rows  # free list

    unique_mut_patients = mutations_df["Patient_ID"].nunique()

    print(f"\n  Total mutation rows loaded   : {total_mutations:,}")
    print(f"  Total chunks read            : {chunks_read}")
    print(f"  Unique patients (mutations)  : {unique_mut_patients:,}")
    validate_patient_ids(mutations_df["Patient_ID"].unique(), "mutation data")

    # ====================================================================
    # STEP 2 — Load TCGA pathology reports
    # ====================================================================
    print("\n" + "=" * 65)
    print("STEP 2: Loading TCGA pathology reports")
    print("=" * 65)

    reports_df = pd.read_csv(REPORTS_FILE)
    # Extract Patient_ID via regex from patient_filename
    reports_df["Patient_ID"] = (
        reports_df["patient_filename"]
        .astype(str)
        .str.extract(PATIENT_ID_EXTRACT_RE, expand=False)
    )
    reports_df = reports_df.dropna(subset=["Patient_ID"])

    unique_rpt_patients = reports_df["Patient_ID"].nunique()

    print(f"  Total report rows            : {len(reports_df):,}")
    print(f"  Unique patients (reports)    : {unique_rpt_patients:,}")
    validate_patient_ids(reports_df["Patient_ID"].unique(), "reports")

    # ====================================================================
    # STEP 3 — Inner merge on Patient_ID
    # ====================================================================
    print("\n" + "=" * 65)
    print("STEP 3: Merging datasets (inner join on Patient_ID)")
    print("=" * 65)

    # For the merge checkpoint we want one row per patient with their
    # report text + summary mutation info.  But the user also wants the
    # full per-mutation table saved, so we do both.

    # 3a. Full mutation table filtered to matched patients only
    matched_patients = set(mutations_df["Patient_ID"]) & set(reports_df["Patient_ID"])
    mutations_matched = mutations_df[
        mutations_df["Patient_ID"].isin(matched_patients)
    ].copy()

    # 3b. Merged checkpoint: one row per patient (report + mutation count)
    patient_mut_counts = (
        mutations_matched
        .groupby("Patient_ID")
        .agg(
            mutation_count=("Hugo_Symbol", "size"),
            unique_genes=("Hugo_Symbol", "nunique"),
            variant_types=("Variant_Classification", lambda x: "|".join(sorted(x.unique()))),
        )
        .reset_index()
    )

    # Keep one report row per patient (deduplicate)
    reports_dedup = reports_df.drop_duplicates(subset=["Patient_ID"])

    merged_df = reports_dedup.merge(patient_mut_counts, on="Patient_ID", how="inner")

    final_patients = merged_df["Patient_ID"].nunique()
    only_in_mut = unique_mut_patients - final_patients
    only_in_rpt = unique_rpt_patients - final_patients

    print(f"  Matched patients             : {final_patients:,}")
    print(f"  Only in mutation data        : {only_in_mut:,}")
    print(f"  Only in reports              : {only_in_rpt:,}")
    print(f"  Mutation rows after filter   : {len(mutations_matched):,}")
    print(f"  Merged checkpoint rows       : {len(merged_df):,}")

    # ====================================================================
    # STEP 4 — Cancer cohort breakdown (TCGA-XX prefix)
    # ====================================================================
    print("\n" + "=" * 65)
    print("STEP 4: Cancer cohort breakdown (matched patients)")
    print("=" * 65)

    merged_df["Cohort"] = merged_df["Patient_ID"].str[:7]  # e.g. TCGA-05
    cohort_counts = (
        merged_df["Cohort"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print(f"\n  {'Cohort':<12} {'Patients':>10}")
    print(f"  {'-'*12} {'-'*10}")
    for cohort, count in cohort_counts.items():
        print(f"  {cohort:<12} {count:>10,}")
    print(f"  {'-'*12} {'-'*10}")
    print(f"  {'TOTAL':<12} {cohort_counts.sum():>10,}")

    # ====================================================================
    # STEP 5 — Save checkpoints
    # ====================================================================
    print("\n" + "=" * 65)
    print("STEP 5: Saving checkpoints")
    print("=" * 65)

    mut_path = os.path.join(OUTPUT_DIR, "mutations_processed.tsv")
    mrg_path = os.path.join(OUTPUT_DIR, "merged_checkpoint.tsv")

    mutations_matched.to_csv(mut_path, sep="\t", index=False)
    merged_df.to_csv(mrg_path, sep="\t", index=False)

    print(f"  -> {mut_path}  ({os.path.getsize(mut_path)/1e6:.1f} MB)")
    print(f"  -> {mrg_path}  ({os.path.getsize(mrg_path)/1e6:.1f} MB)")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print("               ALIGNMENT REPORT")
    print("=" * 65)
    print(f"  Total mutations loaded       : {total_mutations:>10,}")
    print(f"  Total report patients        : {unique_rpt_patients:>10,}")
    print(f"  Mutation patients            : {unique_mut_patients:>10,}")
    print(f"  Final matched patients       : {final_patients:>10,}")
    print(f"  Only in mutation data        : {only_in_mut:>10,}")
    print(f"  Only in reports              : {only_in_rpt:>10,}")
    print(f"  Cancer cohorts represented   : {cohort_counts.nunique():>10,}")
    print(f"  Time elapsed                 : {elapsed:>9.1f}s")
    print("=" * 65)
    print("\nCheckpoints saved -- you never need to re-load the 3.5 GB MAF again.")


if __name__ == "__main__":
    main()
