"""
Feature Engineering Pipeline (Phase 3)
=======================================
Merges mutation labels with pathology reports, cleans text,
extracts TF-IDF features, encodes labels, and saves all
artifacts for downstream ML training.

Input:
  - output/patient_mutation_labels.tsv
  - TCGA_Reports.csv
Output:
  - output/X_tfidf.npz, y_3class.npy, y_binary.npy
  - output/patient_ids.npy, tfidf_vectorizer.pkl, clean_reports.tsv
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ── Paths ───────────────────────────────────────────────────────────────
LABELS_FILE = "output/patient_mutation_labels.tsv"
REPORTS_FILE = "TCGA_Reports.csv"
OUTPUT_DIR = "output"

PATIENT_ID_RE = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)")

# ── Medical abbreviation expansions ────────────────────────────────────
MEDICAL_ABBREVS = {
    r"\badenoca\b": "adenocarcinoma",
    r"\bscc\b": "squamous cell carcinoma",
    r"\bmets\b": "metastasis",
    r"\blvi\b": "lymphovascular invasion",
    r"\bdz\b": "disease",
    r"\bhx\b": "history",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\bpni\b": "perineural invasion",
    r"\bca\b": "carcinoma",
    r"\bbx\b": "biopsy",
    r"\bpathol\b": "pathological",
    r"\brt\b": "radiation therapy",
    r"\bchemo\b": "chemotherapy",
    r"\bneoadj\b": "neoadjuvant",
    r"\badj\b": "adjuvant",
    r"\bhem\b": "hematoxylin",
}

# ── Boilerplate phrases to strip ───────────────────────────────────────
BOILERPLATE_PATTERNS = [
    r"report generated.*",
    r"electronic signature.*",
    r"electronically signed.*",
    r"this report was generated.*",
    r"end of report.*",
    r"pathology report.*page \d+.*",
    r"printed on.*",
    r"disclaimer:.*",
    r"this is a confidential.*",
    r"copy\s*no\.?\s*\d+",
    r"page\s*\d+\s*/\s*\d+",
]

# ── Label encodings ────────────────────────────────────────────────────
LABEL_3CLASS = {
    "Targetable_Mutation": 2,
    "Non_Targetable_Mutation": 1,
    "No_Mutation": 0,
}
LABEL_BINARY = {
    "Targetable_Mutation": 1,
    "Non_Targetable_Mutation": 0,
    "No_Mutation": 0,
}


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def clean_text(text):
    """
    Clean a single pathology report:
    1. Lowercase
    2. Expand medical abbreviations
    3. Remove boilerplate
    4. Remove special characters (keep hyphens, periods)
    5. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Expand medical abbreviations
    for pattern, expansion in MEDICAL_ABBREVS.items():
        text = re.sub(pattern, expansion, text)

    # Remove boilerplate
    for bp in BOILERPLATE_PATTERNS:
        text = re.sub(bp, "", text, flags=re.IGNORECASE)

    # Remove special characters but keep hyphens, periods, and spaces
    # Hyphens in medical terms (e.g., well-differentiated)
    # Periods in measurements (e.g., 2.5 cm)
    text = re.sub(r"[^a-z0-9\s.\-]", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_top_unigrams(texts, n=20):
    """Return the top-n unigrams by raw frequency."""
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    return counter.most_common(n)


def main():
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # STEP 1 -- Merge labels with reports
    # ================================================================
    section("STEP 1: Merging labels with reports")

    labels = pd.read_csv(LABELS_FILE, sep="\t")
    reports = pd.read_csv(REPORTS_FILE)

    print(f"  Labels loaded            : {len(labels):>8,} patients")
    print(f"  Reports loaded           : {len(reports):>8,} rows")

    # Extract Patient_ID from patient_filename
    reports["Patient_ID"] = (
        reports["patient_filename"]
        .astype(str)
        .str.extract(PATIENT_ID_RE, expand=False)
    )
    reports = reports.dropna(subset=["Patient_ID"])

    # Deduplicate reports: keep longest report per patient
    reports["text_len"] = reports["text"].astype(str).str.len()
    reports = (
        reports
        .sort_values("text_len", ascending=False)
        .drop_duplicates(subset=["Patient_ID"], keep="first")
        .drop(columns=["text_len"])
    )
    print(f"  Unique report patients   : {reports['Patient_ID'].nunique():>8,}")

    # Inner merge
    merged = labels.merge(reports[["Patient_ID", "text"]], on="Patient_ID", how="inner")
    print(f"  After inner merge        : {len(merged):>8,}")

    # Drop null / empty text
    merged["text"] = merged["text"].astype(str)
    merged = merged[merged["text"].str.strip().str.len() > 0].copy()
    merged = merged[merged["text"] != "nan"].copy()
    print(f"  After dropping empty text: {len(merged):>8,}")

    print(f"\n  Class distribution (merged):")
    for cls, cnt in merged["Mutation_Status"].value_counts().items():
        pct = cnt / len(merged) * 100
        print(f"    {cls:<30} {cnt:>6,}  ({pct:.1f}%)")

    # ================================================================
    # STEP 2 -- Text cleaning
    # ================================================================
    section("STEP 2: Cleaning pathology reports")

    # Store originals for before/after demo
    sample_indices = merged.sample(n=3, random_state=42).index

    merged["clean_text"] = merged["text"].apply(clean_text)

    # Drop any that become empty after cleaning
    empty_after = (merged["clean_text"].str.strip().str.len() == 0).sum()
    merged = merged[merged["clean_text"].str.strip().str.len() > 0].copy()
    print(f"  Reports cleaned          : {len(merged):>8,}")
    print(f"  Empty after cleaning     : {empty_after:>8,}")

    # Text length statistics
    text_lens = merged["clean_text"].str.split().str.len()
    print(f"\n  Clean text word count stats:")
    print(f"    Mean                   : {text_lens.mean():>10.0f}")
    print(f"    Median                 : {text_lens.median():>10.0f}")
    print(f"    Min                    : {text_lens.min():>10}")
    print(f"    Max                    : {text_lens.max():>10}")

    # Before / after samples
    print(f"\n  --- Before / After Cleaning (3 samples) ---")
    for i, idx in enumerate(sample_indices, 1):
        if idx not in merged.index:
            continue
        row = merged.loc[idx]
        orig = row["text"][:200]
        cleaned = row["clean_text"][:200]
        print(f"\n  Sample {i} ({row['Patient_ID']}, {row['Mutation_Status']}):")
        print(f"    BEFORE: {orig}...")
        print(f"    AFTER : {cleaned}...")

    # ================================================================
    # STEP 3 -- TF-IDF feature extraction
    # ================================================================
    section("STEP 3: TF-IDF feature extraction")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        analyzer="word",
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-z][a-z0-9.\-]{1,}\b",  # at least 2 chars, starts with letter
    )

    X_tfidf = vectorizer.fit_transform(merged["clean_text"])

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vocabulary size          : {vocab_size:>8,}")
    print(f"  Feature matrix shape     : {X_tfidf.shape}")
    print(f"  Non-zero entries         : {X_tfidf.nnz:>12,}")
    print(f"  Sparsity                 : {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4%}")
    print(f"  Matrix size (MB)         : {(X_tfidf.data.nbytes + X_tfidf.indices.nbytes + X_tfidf.indptr.nbytes) / 1e6:.1f}")

    # ================================================================
    # STEP 4 -- Label encoding
    # ================================================================
    section("STEP 4: Label encoding")

    y_3class = merged["Mutation_Status"].map(LABEL_3CLASS).values
    y_binary = merged["Mutation_Status"].map(LABEL_BINARY).values
    patient_ids = merged["Patient_ID"].values

    # 3-class distribution
    print(f"\n  3-Class encoding (Targetable=2, Non_Targetable=1, No_Mutation=0):")
    for label_name, label_val in LABEL_3CLASS.items():
        cnt = (y_3class == label_val).sum()
        pct = cnt / len(y_3class) * 100
        print(f"    {label_name:<30} (={label_val})  {cnt:>6,}  ({pct:.1f}%)")

    # Binary distribution
    print(f"\n  Binary encoding (Targetable=1, else=0):")
    for val, name in [(1, "Targetable (positive)"), (0, "Non-Targetable + None (negative)")]:
        cnt = (y_binary == val).sum()
        pct = cnt / len(y_binary) * 100
        print(f"    {name:<40}  {cnt:>6,}  ({pct:.1f}%)")

    # ================================================================
    # STEP 5 -- Class imbalance analysis
    # ================================================================
    section("STEP 5: Class imbalance analysis")

    # 3-class
    counts_3 = np.bincount(y_3class)
    max_3, min_3 = counts_3.max(), counts_3[counts_3 > 0].min()
    ratio_3 = max_3 / min_3
    print(f"\n  3-Class imbalance:")
    print(f"    Majority class count   : {max_3:>8,}")
    print(f"    Minority class count   : {min_3:>8,}")
    print(f"    Imbalance ratio        : {ratio_3:>8.1f}:1")

    # Binary
    pos = (y_binary == 1).sum()
    neg = (y_binary == 0).sum()
    ratio_bin = max(pos, neg) / min(pos, neg)
    print(f"\n  Binary imbalance:")
    print(f"    Positive (Targetable)  : {pos:>8,}")
    print(f"    Negative (Other)       : {neg:>8,}")
    print(f"    Imbalance ratio        : {ratio_bin:>8.1f}:1")

    # Recommendations
    print(f"\n  RECOMMENDATIONS:")
    if ratio_3 > 20:
        print(f"    3-class: Severe imbalance ({ratio_3:.0f}:1).")
        print(f"      -> Use class_weight='balanced' + stratified CV")
        print(f"      -> Consider SMOTE for minority class, or merge No_Mutation into Non_Targetable")
    elif ratio_3 > 5:
        print(f"    3-class: Moderate imbalance ({ratio_3:.0f}:1).")
        print(f"      -> Use class_weight='balanced' in all models")
        print(f"      -> SMOTE optional but recommended for the minority class")
    else:
        print(f"    3-class: Mild imbalance ({ratio_3:.1f}:1).")
        print(f"      -> class_weight='balanced' should suffice")

    if ratio_bin > 5:
        print(f"    Binary:  Moderate imbalance ({ratio_bin:.1f}:1).")
        print(f"      -> Use class_weight='balanced'")
    else:
        print(f"    Binary:  Acceptable imbalance ({ratio_bin:.1f}:1).")
        print(f"      -> class_weight='balanced' sufficient, SMOTE not needed")

    # ================================================================
    # STEP 6 -- Top 20 unigrams per class
    # ================================================================
    section("STEP 6: Top 20 unigrams by class")

    for cls_name in ["Targetable_Mutation", "Non_Targetable_Mutation"]:
        mask = merged["Mutation_Status"] == cls_name
        top = get_top_unigrams(merged.loc[mask, "clean_text"], n=20)
        print(f"\n  {cls_name} ({mask.sum():,} reports):")
        print(f"    {'Rank':<6} {'Unigram':<25} {'Count':>10}")
        print(f"    {'-'*6} {'-'*25} {'-'*10}")
        for rank, (word, cnt) in enumerate(top, 1):
            print(f"    {rank:<6} {word:<25} {cnt:>10,}")

    # ================================================================
    # STEP 7 -- Save all artifacts
    # ================================================================
    section("STEP 7: Saving artifacts")

    # Sanity check
    assert X_tfidf.shape[0] == len(y_3class) == len(y_binary) == len(patient_ids), \
        f"MISMATCH: X={X_tfidf.shape[0]}, y3={len(y_3class)}, yb={len(y_binary)}, ids={len(patient_ids)}"
    print(f"  [OK] X, y, patient_ids all have {X_tfidf.shape[0]:,} samples")

    # Save sparse matrix
    path_X = os.path.join(OUTPUT_DIR, "X_tfidf.npz")
    sparse.save_npz(path_X, X_tfidf)
    print(f"  Saved: {path_X:<40} ({os.path.getsize(path_X)/1e6:.1f} MB)")

    # Save label arrays
    path_y3 = os.path.join(OUTPUT_DIR, "y_3class.npy")
    np.save(path_y3, y_3class)
    print(f"  Saved: {path_y3:<40} ({os.path.getsize(path_y3)/1e6:.2f} MB)")

    path_yb = os.path.join(OUTPUT_DIR, "y_binary.npy")
    np.save(path_yb, y_binary)
    print(f"  Saved: {path_yb:<40} ({os.path.getsize(path_yb)/1e6:.2f} MB)")

    # Save patient IDs
    path_ids = os.path.join(OUTPUT_DIR, "patient_ids.npy")
    np.save(path_ids, patient_ids)
    print(f"  Saved: {path_ids:<40} ({os.path.getsize(path_ids)/1e6:.2f} MB)")

    # Save fitted vectorizer
    path_vec = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    with open(path_vec, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved: {path_vec:<40} ({os.path.getsize(path_vec)/1e6:.1f} MB)")

    # Save clean reports TSV
    path_clean = os.path.join(OUTPUT_DIR, "clean_reports.tsv")
    merged[["Patient_ID", "clean_text", "Mutation_Status"]].to_csv(
        path_clean, sep="\t", index=False
    )
    print(f"  Saved: {path_clean:<40} ({os.path.getsize(path_clean)/1e6:.1f} MB)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start
    section("FEATURE ENGINEERING COMPLETE")

    print(f"""
  Patients in final dataset    : {len(merged):>8,}
  TF-IDF feature dimensions   : {X_tfidf.shape[1]:>8,}
  Vocabulary size              : {vocab_size:>8,}

  3-Class distribution:
    Targetable_Mutation        : {(y_3class == 2).sum():>8,}
    Non_Targetable_Mutation    : {(y_3class == 1).sum():>8,}
    No_Mutation                : {(y_3class == 0).sum():>8,}

  Binary distribution:
    Positive (Targetable)      : {(y_binary == 1).sum():>8,}
    Negative (Other)           : {(y_binary == 0).sum():>8,}

  Artifacts saved to: {OUTPUT_DIR}/
  Time elapsed: {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
