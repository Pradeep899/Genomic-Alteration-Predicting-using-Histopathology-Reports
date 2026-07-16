"""
Clinical Analysis & Final Report Pipeline (Phase 5)
=====================================================
Generates clinical output tables, per-gene analysis, cancer-type
stratified performance, error analysis, and summary statistics.
"""

import os, sys, pickle, warnings, numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "output"

# FDA-approved drug mappings
DRUG_MAP = {
    "EGFR": "Osimertinib", "BRAF": "Vemurafenib", "KRAS": "Sotorasib",
    "ALK": "Alectinib", "BRCA1": "Olaparib", "BRCA2": "Olaparib",
    "PIK3CA": "Alpelisib", "ERBB2": "Trastuzumab", "IDH1": "Ivosidenib",
    "IDH2": "Enasidenib", "FGFR1": "Erdafitinib", "FGFR2": "Erdafitinib",
    "FGFR3": "Erdafitinib", "FGFR4": "Erdafitinib", "MET": "Capmatinib",
    "RET": "Selpercatinib", "ROS1": "Crizotinib", "NTRK1": "Larotrectinib",
    "NTRK2": "Larotrectinib", "NTRK3": "Larotrectinib", "KIT": "Imatinib",
    "PDGFRA": "Avapritinib", "PTEN": "Ipatasertib (investigational)",
    "ATM": "Olaparib (investigational)", "ERBB3": "Pertuzumab",
    "CDK4": "Palbociclib", "CDK6": "Palbociclib", "FLT3": "Midostaurin",
    "JAK2": "Ruxolitinib", "ABL1": "Imatinib", "SMO": "Vismodegib",
    "MTOR": "Everolimus", "VHL": "Belzutifan",
}

FOLLOW_UP = {
    "Targetable_Mutation": "URGENT: Refer to oncologist -- targeted therapy indicated",
    "Non_Targetable_Mutation": "MONITOR: Standard-of-care; consider clinical trial",
    "No_Mutation": "ROUTINE: Standard surveillance protocol",
}

def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def main():
    # ── Load all data ───────────────────────────────────────────────────
    section("Loading data & model")
    X = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_tfidf.npz"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_binary.npy"))
    ids = np.load(os.path.join(OUTPUT_DIR, "patient_ids.npy"), allow_pickle=True)
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    labels_df = pd.read_csv(os.path.join(OUTPUT_DIR, "patient_mutation_labels.tsv"), sep="\t")
    clean_df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_reports.tsv"), sep="\t")

    print(f"  Patients: {X.shape[0]:,}  |  Features: {X.shape[1]:,}")
    print(f"  Model: {type(model).__name__}")

    # ════════════════════════════════════════════════════════════════════
    # 1. PREDICTIONS FOR ALL PATIENTS
    # ════════════════════════════════════════════════════════════════════
    section("STEP 1: Generating predictions for all patients")

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    pred_df = pd.DataFrame({
        "Patient_ID": ids,
        "True_Label": y,
        "Predicted_Class": np.where(y_pred == 1, "Targetable_Mutation", "Non_Targetable"),
        "Confidence_Score": np.round(y_prob, 4),
        "Correct": y == y_pred,
    })

    acc = pred_df["Correct"].mean()
    auc = roc_auc_score(y, y_prob)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"  Overall AUC         : {auc:.4f}")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity         : {specificity:.4f}")
    print(f"  True Pos: {tp:,}  False Pos: {fp:,}  True Neg: {tn:,}  False Neg: {fn:,}")

    # ════════════════════════════════════════════════════════════════════
    # 2. FINAL CLINICAL OUTPUT TABLE
    # ════════════════════════════════════════════════════════════════════
    section("STEP 2: Building final clinical report")

    # Merge predictions with mutation labels
    report = pred_df[["Patient_ID", "Predicted_Class", "Confidence_Score"]].merge(
        labels_df[["Patient_ID", "Mutation_Status", "targetable_genes", "n_mutations"]],
        on="Patient_ID", how="left"
    )

    report["Follow_Up_Decision"] = report["Mutation_Status"].map(FOLLOW_UP)

    # Priority score: targetable status (0.5) + confidence (0.3) + n_mutations scaled (0.2)
    max_mut = report["n_mutations"].max()
    report["Priority_Score"] = np.round(
        (report["Mutation_Status"] == "Targetable_Mutation").astype(float) * 0.5
        + report["Confidence_Score"] * 0.3
        + (report["n_mutations"] / max_mut) * 0.2, 4
    )

    # Low confidence flag
    low_conf_mask = report["Confidence_Score"].between(0.4, 0.6)
    report.loc[low_conf_mask, "Follow_Up_Decision"] = (
        report.loc[low_conf_mask, "Follow_Up_Decision"]
        + " | LOW CONFIDENCE -- manual review recommended"
    )

    # Sort: targetable first, then confidence descending
    status_order = {"Targetable_Mutation": 0, "Non_Targetable_Mutation": 1, "No_Mutation": 2}
    report["_sort"] = report["Mutation_Status"].map(status_order)
    report = report.sort_values(["_sort", "Confidence_Score"], ascending=[True, False])
    report = report.drop(columns=["_sort"])

    # Rename for clinical readability
    report_out = report.rename(columns={
        "Mutation_Status": "True_Mutation_Status",
        "targetable_genes": "Targetable_Genes",
        "n_mutations": "N_Mutations",
        "Confidence_Score": "Confidence",
    })

    p1 = os.path.join(OUTPUT_DIR, "final_patient_report.tsv")
    report_out.to_csv(p1, sep="\t", index=False)
    print(f"  Saved: {p1} ({len(report_out):,} patients)")
    print(f"  Low-confidence cases flagged: {low_conf_mask.sum():,}")
    print(f"  Sample (top 5):")
    print(report_out[["Patient_ID","True_Mutation_Status","Predicted_Class","Confidence","Priority_Score"]].head().to_string(index=False))

    # ════════════════════════════════════════════════════════════════════
    # 3. PER-GENE FREQUENCY ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    section("STEP 3: Per-gene frequency analysis")

    # Explode targetable genes
    gene_df = labels_df[labels_df["targetable_genes"].notna() & (labels_df["targetable_genes"] != "")].copy()
    gene_df["gene_list"] = gene_df["targetable_genes"].str.split("|")
    exploded = gene_df.explode("gene_list")

    # Add cohort (cancer type proxy = TCGA-XX)
    exploded["Cancer_Type"] = exploded["Patient_ID"].str[:7]

    total_patients = len(labels_df)
    gene_stats = []
    for gene, grp in exploded.groupby("gene_list"):
        n_pts = grp["Patient_ID"].nunique()
        top_cancers = grp["Cancer_Type"].value_counts().head(3)
        top3_str = ", ".join([f"{ct}({cnt})" for ct, cnt in top_cancers.items()])
        drug = DRUG_MAP.get(gene, "No FDA-approved targeted therapy")
        gene_stats.append({
            "Gene": gene,
            "N_Patients_Mutated": n_pts,
            "Pct_of_Cohort": round(n_pts / total_patients * 100, 2),
            "Top_3_Cancer_Types": top3_str,
            "Recommended_Drug": drug,
        })

    gene_report = pd.DataFrame(gene_stats).sort_values("N_Patients_Mutated", ascending=False)
    p2 = os.path.join(OUTPUT_DIR, "gene_frequency_report.tsv")
    gene_report.to_csv(p2, sep="\t", index=False)
    print(f"  Saved: {p2} ({len(gene_report)} genes)")
    print(f"\n  Top 10 targetable genes:")
    print(f"    {'Gene':<12} {'Patients':>10} {'%':>7}  Drug")
    print(f"    {'-'*12} {'-'*10} {'-'*7}  {'-'*25}")
    for _, r in gene_report.head(10).iterrows():
        print(f"    {r['Gene']:<12} {r['N_Patients_Mutated']:>10,} {r['Pct_of_Cohort']:>6.1f}%  {r['Recommended_Drug']}")

    # ════════════════════════════════════════════════════════════════════
    # 4. CANCER-TYPE STRATIFIED PERFORMANCE
    # ════════════════════════════════════════════════════════════════════
    section("STEP 4: Cancer-type stratified performance")

    pred_df["Cancer_Type"] = pred_df["Patient_ID"].str[:7]
    cancer_results = []
    for ct, grp in pred_df.groupby("Cancer_Type"):
        if len(grp) < 10:
            continue
        n_tgt = int((grp["True_Label"] == 1).sum())
        n_non = int((grp["True_Label"] == 0).sum())
        # Need both classes for AUC
        if n_tgt == 0 or n_non == 0:
            auc_ct = float("nan")
        else:
            auc_ct = roc_auc_score(grp["True_Label"], grp["Confidence_Score"])
        f1_ct = f1_score(grp["True_Label"], (grp["Confidence_Score"] >= 0.5).astype(int), zero_division=0)
        cancer_results.append({
            "Cancer_Type": ct, "N_Patients": len(grp),
            "N_Targetable": n_tgt, "N_Non_Targetable": n_non,
            "Targetable_Rate": round(n_tgt / len(grp) * 100, 1),
            "AUC": round(auc_ct, 4) if not np.isnan(auc_ct) else "N/A",
            "F1_Targetable": round(f1_ct, 4),
        })

    cancer_df = pd.DataFrame(cancer_results).sort_values("N_Patients", ascending=False)
    p3 = os.path.join(OUTPUT_DIR, "per_cancer_results.tsv")
    cancer_df.to_csv(p3, sep="\t", index=False)
    print(f"  Saved: {p3} ({len(cancer_df)} cancer types)")
    print(f"\n  Top 15 cancer types by size:")
    print(f"    {'Type':<10} {'N':>6} {'Tgt%':>6} {'AUC':>8} {'F1':>8}")
    print(f"    {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for _, r in cancer_df.head(15).iterrows():
        print(f"    {r['Cancer_Type']:<10} {r['N_Patients']:>6} {r['Targetable_Rate']:>5.1f}% {str(r['AUC']):>8} {r['F1_Targetable']:>8.4f}")

    # ════════════════════════════════════════════════════════════════════
    # 5. ERROR ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    section("STEP 5: Error analysis")

    # Merge predictions with labels and clean text
    error_df = pred_df.merge(
        labels_df[["Patient_ID", "Mutation_Status", "targetable_genes"]],
        on="Patient_ID", how="left"
    ).merge(
        clean_df[["Patient_ID", "clean_text"]], on="Patient_ID", how="left"
    )

    # False Negatives: truly targetable but predicted as non-targetable
    fn_df = error_df[(error_df["True_Label"] == 1) & (y_pred[error_df.index] == 0)].copy()
    fn_df["Error_Type"] = "FALSE_NEGATIVE (Targetable missed)"
    fn_df["Report_Excerpt"] = fn_df["clean_text"].str[:200]

    # False Positives: truly non-targetable but predicted as targetable
    fp_df = error_df[(error_df["True_Label"] == 0) & (y_pred[error_df.index] == 1)].copy()
    fp_df["Error_Type"] = "FALSE_POSITIVE (Over-referred)"
    fp_df["Report_Excerpt"] = fp_df["clean_text"].str[:200]

    misclass = pd.concat([fn_df, fp_df], ignore_index=True)
    cols_out = ["Patient_ID", "Error_Type", "Mutation_Status", "targetable_genes",
                "Confidence_Score", "Cancer_Type", "Report_Excerpt"]
    misclass_out = misclass[[c for c in cols_out if c in misclass.columns]]

    p4 = os.path.join(OUTPUT_DIR, "misclassified_cases.tsv")
    misclass_out.to_csv(p4, sep="\t", index=False)

    print(f"  False Negatives (missed targetable): {len(fn_df):,}")
    print(f"  False Positives (over-referred)    : {len(fp_df):,}")
    print(f"  Total misclassified                : {len(misclass):,}")
    print(f"  Saved: {p4}")

    # Show worst FN examples (highest confidence in wrong class)
    if len(fn_df) > 0:
        print(f"\n  CRITICAL: Top 5 False Negatives (missed targetable mutations):")
        for _, r in fn_df.sort_values("Confidence_Score").head(5).iterrows():
            genes = r.get("targetable_genes", "")
            print(f"    {r['Patient_ID']} | conf={r['Confidence_Score']:.3f} | genes={genes}")

    # ════════════════════════════════════════════════════════════════════
    # 6. SUMMARY STATISTICS REPORT
    # ════════════════════════════════════════════════════════════════════
    section("STEP 6: Summary statistics report")

    top5_genes = gene_report.head(5)
    n_benefit = int((labels_df["Mutation_Status"] == "Targetable_Mutation").sum())
    pct_benefit = n_benefit / total_patients * 100

    # Cancer types with highest targetable rates (min 20 patients)
    top_cancers_tgt = cancer_df[cancer_df["N_Patients"] >= 20].sort_values(
        "Targetable_Rate", ascending=False
    ).head(10)

    summary_lines = [
        "=" * 70,
        "  CLINICAL GENOMICS PREDICTION -- SUMMARY REPORT",
        "  Project: Predicting Targetable Prognostic Genomic Alterations",
        "           Using Histopathology Reports",
        "=" * 70,
        "",
        "COHORT OVERVIEW",
        f"  Total patients processed          : {total_patients:,}",
        f"  Patients with targetable mutations: {n_benefit:,} ({pct_benefit:.1f}%)",
        f"  Patients without targetable muts  : {total_patients - n_benefit:,} ({100 - pct_benefit:.1f}%)",
        f"  Unique cancer types (TCGA cohort) : {pred_df['Cancer_Type'].nunique()}",
        "",
        "MODEL PERFORMANCE",
        f"  Model type                        : {type(model).__name__}",
        f"  AUC-ROC                           : {auc:.4f}",
        f"  Sensitivity (True Positive Rate)  : {sensitivity:.4f}",
        f"  Specificity (True Negative Rate)  : {specificity:.4f}",
        f"  False Negative Rate               : {1 - sensitivity:.4f}  (missed targetable)",
        f"  False Positive Rate               : {1 - specificity:.4f}  (over-referred)",
        "",
        "CLINICAL IMPACT",
        f"  Estimated patients benefiting from targeted therapy: {n_benefit:,} ({pct_benefit:.1f}%)",
        f"  Patients correctly identified     : {tp:,} / {n_benefit:,} ({tp/n_benefit*100:.1f}%)",
        f"  Patients missed (False Negatives) : {fn:,} -- REQUIRES attention",
        f"  Unnecessary referrals (False Pos) : {fp:,}",
        f"  Low-confidence predictions        : {low_conf_mask.sum():,} (manual review flagged)",
        "",
        "TOP 5 MOST ACTIONABLE GENES IN COHORT",
    ]
    for _, r in top5_genes.iterrows():
        summary_lines.append(
            f"  {r['Gene']:<10} {r['N_Patients_Mutated']:>5,} patients ({r['Pct_of_Cohort']:.1f}%)  "
            f"-> {r['Recommended_Drug']}"
        )

    summary_lines += [
        "",
        "TARGETABLE MUTATION PREVALENCE BY CANCER TYPE (top 10, n>=20)",
    ]
    for _, r in top_cancers_tgt.iterrows():
        summary_lines.append(f"  {r['Cancer_Type']:<10} {r['Targetable_Rate']:>5.1f}% targetable  (n={r['N_Patients']})")

    summary_lines += [
        "",
        "OUTPUT FILES",
        f"  final_patient_report.tsv    -- Clinical decision table ({total_patients:,} patients)",
        f"  gene_frequency_report.tsv   -- Per-gene analysis ({len(gene_report)} genes)",
        f"  per_cancer_results.tsv      -- Cancer-type AUC/F1 ({len(cancer_df)} types)",
        f"  misclassified_cases.tsv     -- Error analysis ({len(misclass):,} cases)",
        f"  summary_report.txt          -- This file",
        "",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    p5 = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(p5, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {p5}")

    section("PIPELINE COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
