"""
Patient Overlap Analysis
========================
Compares Patient IDs between the MC3 mutation MAF file and TCGA Reports CSV.
Uses chunked reading for the large MAF file to avoid memory issues.
"""

import pandas as pd
import time

MAF_FILE = "mc3.v0.2.8.PUBLIC.maf"
REPORTS_FILE = "TCGA_Reports.csv"
CHUNK_SIZE = 500_000  # rows per chunk

def main():
    start = time.time()

    # ── 1. Extract unique Patient_IDs from Reports ──────────────────────
    print("Reading TCGA_Reports.csv ...")
    reports_df = pd.read_csv(REPORTS_FILE, usecols=["patient_filename"])
    # Patient_ID = first 12 characters of patient_filename
    report_patients = set(
        reports_df["patient_filename"].dropna().str[:12].unique()
    )
    print(f"  -> Unique patients in reports: {len(report_patients)}")

    # ── 2. Extract unique Patient_IDs from Mutation file (chunked) ──────
    print(f"\nReading {MAF_FILE} in chunks of {CHUNK_SIZE:,} rows ...")
    mutation_patients = set()
    chunks_read = 0

    reader = pd.read_csv(
        MAF_FILE,
        sep="\t",
        comment="#",
        usecols=["Tumor_Sample_Barcode"],  # only load one column
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        ids = chunk["Tumor_Sample_Barcode"].dropna().str[:12].unique()
        mutation_patients.update(ids)
        chunks_read += 1
        if chunks_read % 5 == 0:
            print(f"  ... processed {chunks_read} chunks  "
                  f"({chunks_read * CHUNK_SIZE:,}+ rows), "
                  f"unique patients so far: {len(mutation_patients)}")

    print(f"  -> Total chunks read: {chunks_read}")
    print(f"  -> Unique patients in mutation file: {len(mutation_patients)}")

    # ── 3. Compute overlap / mismatches ─────────────────────────────────
    common = mutation_patients & report_patients
    only_mutation = mutation_patients - report_patients
    only_reports = report_patients - mutation_patients

    elapsed = time.time() - start

    # ── 4. Print results ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("         PATIENT OVERLAP SUMMARY")
    print("=" * 55)
    print(f"  Mutation patients : {len(mutation_patients):>6,}")
    print(f"  Report patients   : {len(report_patients):>6,}")
    print(f"  Common patients   : {len(common):>6,}")
    print(f"  Only in mutation  : {len(only_mutation):>6,}")
    print(f"  Only in reports   : {len(only_reports):>6,}")
    print("=" * 55)
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print("=" * 55)

    # ── 5. Sample mismatches ────────────────────────────────────────────
    N_SAMPLES = 5

    if only_mutation:
        print(f"\n  Sample patients ONLY in mutation (up to {N_SAMPLES}):")
        for pid in sorted(only_mutation)[:N_SAMPLES]:
            print(f"    - {pid}")

    if only_reports:
        print(f"\n  Sample patients ONLY in reports (up to {N_SAMPLES}):")
        for pid in sorted(only_reports)[:N_SAMPLES]:
            print(f"    - {pid}")

    if not only_mutation and not only_reports:
        print("\n  [OK] Perfect match — all patients are common!")


if __name__ == "__main__":
    main()
