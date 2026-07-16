"""
Mutation Labeling Pipeline
==========================
Classifies each patient's mutation status (Targetable / Non-Targetable / None)
against a literature-backed set of clinically actionable genes.

Input : output/mutations_processed.tsv  (from Phase 1)
Output: output/patient_mutation_labels.tsv
"""

import os
import pandas as pd
import numpy as np

INPUT_FILE = "output/mutations_processed.tsv"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "patient_mutation_labels.tsv")

# ── Biologically meaningful variant types to KEEP ───────────────────────
KEEP_VARIANTS = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Splice_Region",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
}

# ── Hypermutator artefact genes to EXCLUDE ──────────────────────────────
HYPERMUTATOR_GENES = {"TTN", "MUC16", "MUC4", "OBSCN", "XIRP2"}

# ── Literature-backed targetable gene set ───────────────────────────────
TARGETABLE_GENES = {
    # RTK / growth factor receptors
    "EGFR", "ERBB2", "ERBB3", "ALK", "ROS1", "RET",
    "NTRK1", "NTRK2", "NTRK3",
    # MAPK pathway
    "BRAF", "KRAS", "NRAS", "HRAS", "NF1", "MAP2K1", "MAP2K2",
    # FGFR family
    "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    # Other RTKs
    "MET", "KIT", "PDGFRA", "PDGFRB",
    # PI3K / AKT / mTOR
    "PIK3CA", "PIK3R1", "AKT1", "PTEN", "TSC1", "TSC2", "MTOR",
    # Cell cycle / CDK
    "CDK4", "CDK6", "CCND1",
    # DNA damage repair
    "BRCA1", "BRCA2", "ATM", "PALB2", "CDK12", "CHEK2",
    # Metabolic / other targetable
    "IDH1", "IDH2", "FLT3", "VHL", "PBRM1", "BAP1",
    # MMR
    "MLH1", "MSH2", "MSH6", "PMS2",
    # Hedgehog / JAK-STAT / ABL
    "SMO", "PTCH1", "JAK1", "JAK2", "ABL1",
    # Emerging / chromatin
    "POLE", "ARID1A", "SMARCA4", "KDR",
}

HYPERMUTATOR_THRESHOLD = 500


def section(title):
    """Print a section header."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # STEP 0 -- Load raw mutations checkpoint
    # ================================================================
    section("STEP 0: Loading mutations checkpoint")

    df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()

    total_raw = len(df)
    patients_raw = df["Patient_ID"].nunique()
    print(f"  Rows loaded              : {total_raw:>12,}")
    print(f"  Unique patients          : {patients_raw:>12,}")
    print(f"  Unique genes             : {df['Hugo_Symbol'].nunique():>12,}")
    print(f"  Unique variant types     : {df['Variant_Classification'].nunique():>12,}")

    # ================================================================
    # STEP 1 -- Variant filtering
    # ================================================================
    section("STEP 1: Variant filtering (keep biologically meaningful)")

    # Show what we're dropping
    variant_counts = df["Variant_Classification"].value_counts()
    print("\n  Variant type breakdown (before filtering):")
    for vt, cnt in variant_counts.items():
        marker = " [KEEP]" if vt in KEEP_VARIANTS else " [DROP]"
        print(f"    {vt:<30} {cnt:>10,}{marker}")

    df_filtered = df[df["Variant_Classification"].isin(KEEP_VARIANTS)].copy()

    dropped_rows = total_raw - len(df_filtered)
    print(f"\n  Rows before filtering    : {total_raw:>12,}")
    print(f"  Rows after filtering     : {len(df_filtered):>12,}")
    print(f"  Rows dropped             : {dropped_rows:>12,}  "
          f"({dropped_rows / total_raw * 100:.1f}%)")
    print(f"  Patients retained        : {df_filtered['Patient_ID'].nunique():>12,}")

    # ================================================================
    # STEP 2 -- Remove hypermutator artefact genes
    # ================================================================
    section("STEP 2: Removing hypermutator artefact genes")

    artefact_mask = df_filtered["Hugo_Symbol"].isin(HYPERMUTATOR_GENES)
    n_artefact = artefact_mask.sum()

    print(f"\n  Artefact gene hits:")
    for gene in sorted(HYPERMUTATOR_GENES):
        cnt = (df_filtered["Hugo_Symbol"] == gene).sum()
        if cnt > 0:
            print(f"    {gene:<12} {cnt:>8,} mutations removed")

    df_clean = df_filtered[~artefact_mask].copy()

    print(f"\n  Rows before removal      : {len(df_filtered):>12,}")
    print(f"  Rows after removal       : {len(df_clean):>12,}")
    print(f"  Rows dropped             : {n_artefact:>12,}")
    print(f"  Patients retained        : {df_clean['Patient_ID'].nunique():>12,}")

    # ================================================================
    # STEP 3 -- Per-patient grouping & labeling
    # ================================================================
    section("STEP 3: Per-patient mutation profiling")

    # Get the full set of patients (some may have lost all mutations)
    all_patients = df["Patient_ID"].unique()

    # Group by patient
    patient_groups = (
        df_clean
        .groupby("Patient_ID")["Hugo_Symbol"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
        .rename(columns={"Hugo_Symbol": "mutated_genes"})
    )

    # Build label dataframe for ALL patients
    label_df = pd.DataFrame({"Patient_ID": all_patients})
    label_df = label_df.merge(patient_groups, on="Patient_ID", how="left")

    # Fill patients with no remaining mutations
    label_df["mutated_genes"] = label_df["mutated_genes"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Compute fields
    label_df["n_mutations"] = label_df["mutated_genes"].apply(len)
    label_df["targetable_genes"] = label_df["mutated_genes"].apply(
        lambda genes: sorted(set(genes) & TARGETABLE_GENES)
    )
    label_df["n_targetable"] = label_df["targetable_genes"].apply(len)

    # Assign Mutation_Status
    def classify(row):
        if row["n_targetable"] > 0:
            return "Targetable_Mutation"
        elif row["n_mutations"] > 0:
            return "Non_Targetable_Mutation"
        else:
            return "No_Mutation"

    label_df["Mutation_Status"] = label_df.apply(classify, axis=1)

    # Convert lists to pipe-delimited strings for TSV storage
    label_df["targetable_genes_str"] = label_df["targetable_genes"].apply(
        lambda g: "|".join(g) if g else ""
    )
    label_df["all_mutated_genes"] = label_df["mutated_genes"].apply(
        lambda g: "|".join(g) if g else ""
    )

    print(f"  Total patients profiled  : {len(label_df):>12,}")

    # ================================================================
    # STEP 4 -- Class distribution report
    # ================================================================
    section("STEP 4: Class distribution")

    class_counts = label_df["Mutation_Status"].value_counts()
    total_pts = len(label_df)

    print(f"\n  {'Class':<30} {'Count':>8}  {'Pct':>7}")
    print(f"  {'-'*30} {'-'*8}  {'-'*7}")
    for cls in ["Targetable_Mutation", "Non_Targetable_Mutation", "No_Mutation"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / total_pts * 100
        print(f"  {cls:<30} {cnt:>8,}  {pct:>6.1f}%")
    print(f"  {'-'*30} {'-'*8}  {'-'*7}")
    print(f"  {'TOTAL':<30} {total_pts:>8,}  {100.0:>6.1f}%")

    # ================================================================
    # STEP 5 -- Top 20 most mutated targetable genes
    # ================================================================
    section("STEP 5: Top 20 most mutated targetable genes")

    # Explode targetable genes to count patient-level frequency
    tgt_exploded = label_df[label_df["n_targetable"] > 0].explode("targetable_genes")
    gene_patient_counts = (
        tgt_exploded
        .groupby("targetable_genes")["Patient_ID"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
    )

    print(f"\n  {'Rank':<6} {'Gene':<12} {'Patients':>10}  {'% of Matched':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}  {'-'*12}")
    for rank, (gene, cnt) in enumerate(gene_patient_counts.items(), 1):
        pct = cnt / total_pts * 100
        print(f"  {rank:<6} {gene:<12} {cnt:>10,}  {pct:>11.1f}%")

    # ================================================================
    # STEP 6 -- Hypermutator flagging
    # ================================================================
    section("STEP 6: Hypermutator check (n_mutations > 500)")

    hypermut = label_df[label_df["n_mutations"] > HYPERMUTATOR_THRESHOLD]
    print(f"  Patients with > {HYPERMUTATOR_THRESHOLD} mutations : {len(hypermut):,}")

    if len(hypermut) > 0:
        print(f"\n  {'Patient_ID':<16} {'n_mutations':>12} {'n_targetable':>13}  Status")
        print(f"  {'-'*16} {'-'*12} {'-'*13}  {'-'*25}")
        for _, row in hypermut.sort_values("n_mutations", ascending=False).head(15).iterrows():
            print(f"  {row['Patient_ID']:<16} {row['n_mutations']:>12,} "
                  f"{row['n_targetable']:>13}  {row['Mutation_Status']}")
        if len(hypermut) > 15:
            print(f"  ... and {len(hypermut) - 15} more")
    else:
        print("  None found.")

    # ================================================================
    # STEP 7 -- Save output
    # ================================================================
    section("STEP 7: Saving patient mutation labels")

    out_df = label_df[[
        "Patient_ID",
        "Mutation_Status",
        "targetable_genes_str",
        "n_mutations",
        "all_mutated_genes",
    ]].rename(columns={"targetable_genes_str": "targetable_genes"})

    out_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    fsize = os.path.getsize(OUTPUT_FILE) / 1e6

    print(f"  Saved: {OUTPUT_FILE}  ({fsize:.1f} MB)")
    print(f"  Rows : {len(out_df):,}")
    print(f"  Cols : {list(out_df.columns)}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    section("PIPELINE COMPLETE")

    print(f"""
  Input mutations              : {total_raw:>10,} rows
  After variant filtering      : {len(df_filtered):>10,} rows  (-{dropped_rows:,})
  After artefact gene removal  : {len(df_clean):>10,} rows  (-{n_artefact:,})
  Patients profiled            : {len(label_df):>10,}
  Targetable gene list size    : {len(TARGETABLE_GENES):>10}
  Hypermutator artefacts       : {len(HYPERMUTATOR_GENES):>10}
  
  CLASS BREAKDOWN:
    Targetable_Mutation        : {class_counts.get("Targetable_Mutation", 0):>8,} patients
    Non_Targetable_Mutation    : {class_counts.get("Non_Targetable_Mutation", 0):>8,} patients
    No_Mutation                : {class_counts.get("No_Mutation", 0):>8,} patients

  Output: {OUTPUT_FILE}
""")


if __name__ == "__main__":
    main()
