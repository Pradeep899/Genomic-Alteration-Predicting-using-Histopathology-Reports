"""
BERT Training & Analysis Pipeline (Phase 6b)
==============================================
Trains classifiers on BERT embeddings, compares with TF-IDF baseline,
runs per-gene multi-label prediction, ensemble model, cancer-type
specific models, and provides an inference function.

Prerequisite: run bert_embeddings.py first to generate X_bert.npy
"""

import os, sys, time, pickle, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                             classification_report, confusion_matrix)
from scipy import sparse

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "output"
BERT_DIR = "bert_models"
RANDOM_STATE = 42

# Top 10 targetable genes for multi-label prediction
TOP_GENES = ["PIK3CA", "PTEN", "ARID1A", "KRAS", "BRAF", "NF1",
             "ATM", "IDH1", "BRCA2", "EGFR"]


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def main():
    start = time.time()
    os.makedirs(BERT_DIR, exist_ok=True)

    # ── Load all data ───────────────────────────────────────────────────
    section("Loading data")

    X_bert_path = os.path.join(OUTPUT_DIR, "X_bert.npy")
    if not os.path.exists(X_bert_path):
        print(f"  [ERROR] {X_bert_path} not found!")
        print(f"  Run bert_embeddings.py first to generate embeddings.")
        sys.exit(1)

    X_bert = np.load(X_bert_path)
    y = np.load(os.path.join(OUTPUT_DIR, "y_binary.npy"))
    ids = np.load(os.path.join(OUTPUT_DIR, "patient_ids.npy"), allow_pickle=True)
    labels_df = pd.read_csv(os.path.join(OUTPUT_DIR, "patient_mutation_labels.tsv"), sep="\t")

    # Load TF-IDF baseline results
    tfidf_results = pd.read_csv(os.path.join(OUTPUT_DIR, "model_results.tsv"), sep="\t")

    print(f"  X_bert shape  : {X_bert.shape}")
    print(f"  y shape       : {y.shape}")
    print(f"  TF-IDF models : {list(tfidf_results['Model'])}")

    # Align labels_df with embedding order
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    labels_df = labels_df[labels_df["Patient_ID"].isin(id_to_idx)]
    labels_df["_idx"] = labels_df["Patient_ID"].map(id_to_idx)
    labels_df = labels_df.sort_values("_idx").reset_index(drop=True)

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: Train/test split (same seed as Phase 4 for fair comparison)
    # ════════════════════════════════════════════════════════════════════
    section("STEP 1: Train/test split")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_bert, y, ids, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: Train 3 models on BERT embeddings
    # ════════════════════════════════════════════════════════════════════
    section("STEP 2: Training classifiers on BERT embeddings")

    models = {}
    bert_results = []

    # Model A: Logistic Regression
    print("\n  [A] Logistic Regression ...")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs",
                            max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    print(f"      Trained in {time.time()-t0:.1f}s")

    # Model B: Random Forest
    print("  [B] Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print(f"      Trained in {time.time()-t0:.1f}s")

    # Model C: SGD
    print("  [C] SGD Classifier ...")
    t0 = time.time()
    sgd = SGDClassifier(loss="modified_huber", penalty="elasticnet",
                        alpha=1e-4, l1_ratio=0.15, class_weight="balanced",
                        max_iter=1000, random_state=RANDOM_STATE)
    sgd.fit(X_train, y_train)
    models["SGD Classifier"] = sgd
    print(f"      Trained in {time.time()-t0:.1f}s")

    # Evaluate
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        test_auc = roc_auc_score(y_test, y_prob)
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        f1 = f1_score(y_test, y_pred)
        bert_results.append({
            "Model": name, "Test_AUC": round(test_auc, 4),
            "CV_AUC_mean": round(cv_scores.mean(), 4),
            "CV_AUC_std": round(cv_scores.std(), 4),
            "F1_Targetable": round(f1, 4),
        })
        print(f"\n  {name}: Test AUC={test_auc:.4f}, CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}, F1={f1:.4f}")

    bert_df = pd.DataFrame(bert_results)

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: Compare BERT vs TF-IDF
    # ════════════════════════════════════════════════════════════════════
    section("STEP 3: BERT vs TF-IDF comparison")

    comparison = []
    for _, br in bert_df.iterrows():
        tfidf_match = tfidf_results[tfidf_results["Model"] == br["Model"]]
        tfidf_auc = tfidf_match["Test_AUC"].values[0] if len(tfidf_match) > 0 else float("nan")
        delta = br["Test_AUC"] - tfidf_auc if not np.isnan(tfidf_auc) else float("nan")
        comparison.append({
            "Model": br["Model"],
            "TF-IDF_AUC": tfidf_auc,
            "BERT_AUC": br["Test_AUC"],
            "Delta": round(delta, 4) if not np.isnan(delta) else "N/A",
            "Winner": "BERT" if delta > 0 else "TF-IDF" if delta < 0 else "TIE",
        })

    comp_df = pd.DataFrame(comparison)
    print(f"\n  {'Model':<25} {'TF-IDF AUC':>11} {'BERT AUC':>10} {'Delta':>8} {'Winner':>8}")
    print(f"  {'-'*25} {'-'*11} {'-'*10} {'-'*8} {'-'*8}")
    for _, r in comp_df.iterrows():
        print(f"  {r['Model']:<25} {r['TF-IDF_AUC']:>11} {r['BERT_AUC']:>10} {str(r['Delta']):>8} {r['Winner']:>8}")

    # Best BERT model
    best_bert_idx = bert_df["Test_AUC"].idxmax()
    best_bert_name = bert_df.loc[best_bert_idx, "Model"]
    best_bert_auc = bert_df.loc[best_bert_idx, "Test_AUC"]
    best_bert_model = models[best_bert_name]
    print(f"\n  >>> Best BERT model: {best_bert_name} (AUC = {best_bert_auc:.4f})")

    # Save best BERT model
    with open(os.path.join(BERT_DIR, "best_bert_model.pkl"), "wb") as f:
        pickle.dump(best_bert_model, f)

    # ════════════════════════════════════════════════════════════════════
    # STEP 4: Ensemble (TF-IDF + BERT soft voting)
    # ════════════════════════════════════════════════════════════════════
    section("STEP 4: Ensemble model (TF-IDF + BERT)")

    # Load TF-IDF model and features for the same test split
    X_tfidf = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_tfidf.npz"))
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
        tfidf_model = pickle.load(f)

    # Must use the SAME split indices
    from sklearn.model_selection import train_test_split as tts
    _, X_tfidf_test, _, y_test_check = tts(
        X_tfidf, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    assert np.array_equal(y_test, y_test_check), "Split mismatch!"

    tfidf_probs = tfidf_model.predict_proba(X_tfidf_test)[:, 1]
    bert_probs = best_bert_model.predict_proba(X_test)[:, 1]

    # Soft voting: average probabilities
    ensemble_probs = (tfidf_probs + bert_probs) / 2
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    tfidf_solo_auc = roc_auc_score(y_test, tfidf_probs)
    bert_solo_auc = roc_auc_score(y_test, bert_probs)

    ensemble_comp = pd.DataFrame([
        {"Model": "TF-IDF (LR)", "AUC": round(tfidf_solo_auc, 4)},
        {"Model": f"BERT ({best_bert_name})", "AUC": round(bert_solo_auc, 4)},
        {"Model": "Ensemble (avg)", "AUC": round(ensemble_auc, 4)},
    ])
    ensemble_comp.to_csv(os.path.join(OUTPUT_DIR, "ensemble_model_comparison.tsv"),
                         sep="\t", index=False)

    print(f"\n  {'Model':<30} {'AUC':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for _, r in ensemble_comp.iterrows():
        print(f"  {r['Model']:<30} {r['AUC']:>8}")
    print(f"\n  Ensemble saved: output/ensemble_model_comparison.tsv")

    # ROC plot: TF-IDF vs BERT vs Ensemble
    fig, ax = plt.subplots(figsize=(8, 7))
    for probs, name, color in [
        (tfidf_probs, f"TF-IDF LR (AUC={tfidf_solo_auc:.4f})", "#4361ee"),
        (bert_probs, f"BERT {best_bert_name} (AUC={bert_solo_auc:.4f})", "#f72585"),
        (ensemble_probs, f"Ensemble (AUC={ensemble_auc:.4f})", "#2ec4b6"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=name)
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC: TF-IDF vs BERT vs Ensemble", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_bert_ensemble.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved: output/roc_bert_ensemble.png")

    # ════════════════════════════════════════════════════════════════════
    # STEP 5: Multi-label per-gene prediction
    # ════════════════════════════════════════════════════════════════════
    section("STEP 5: Per-gene binary prediction (top 10 genes)")

    per_gene_dir = os.path.join(BERT_DIR, "per_gene_models")
    os.makedirs(per_gene_dir, exist_ok=True)

    gene_results = []
    for gene in TOP_GENES:
        # Build binary label: 1 if patient has this gene mutated
        y_gene = labels_df["targetable_genes"].fillna("").str.contains(
            rf"\b{gene}\b", regex=True
        ).astype(int).values

        n_pos = int(y_gene.sum())
        n_neg = int((y_gene == 0).sum())

        if n_pos < 10:
            print(f"  {gene}: too few positives ({n_pos}), skipping")
            continue

        # Same split
        Xg_train, Xg_test, yg_train, yg_test = train_test_split(
            X_bert, y_gene, test_size=0.2, stratify=y_gene, random_state=RANDOM_STATE
        )

        model_g = LogisticRegression(class_weight="balanced", C=1.0,
                                     max_iter=1000, random_state=RANDOM_STATE)
        model_g.fit(Xg_train, yg_train)

        yg_prob = model_g.predict_proba(Xg_test)[:, 1]
        yg_pred = model_g.predict(Xg_test)

        auc_g = roc_auc_score(yg_test, yg_prob) if yg_test.sum() > 0 else float("nan")
        f1_g = f1_score(yg_test, yg_pred, zero_division=0)

        gene_results.append({
            "Gene": gene, "N_Positive": n_pos, "N_Negative": n_neg,
            "AUC": round(auc_g, 4), "F1": round(f1_g, 4),
        })

        # Save gene model
        with open(os.path.join(per_gene_dir, f"{gene}_model.pkl"), "wb") as f:
            pickle.dump(model_g, f)

        print(f"  {gene:<10} pos={n_pos:>5}  neg={n_neg:>5}  AUC={auc_g:.4f}  F1={f1_g:.4f}")

    gene_df = pd.DataFrame(gene_results)
    gene_df.to_csv(os.path.join(OUTPUT_DIR, "per_gene_auc.tsv"), sep="\t", index=False)
    print(f"\n  Saved: output/per_gene_auc.tsv")
    print(f"  Models: {per_gene_dir}/ ({len(gene_results)} genes)")

    # ════════════════════════════════════════════════════════════════════
    # STEP 6: Cancer-type specific models (LUAD, BRCA)
    # ════════════════════════════════════════════════════════════════════
    section("STEP 6: Cancer-type specific models")

    # Map Patient_ID prefix to approximate cancer types
    # TCGA-55, TCGA-05, TCGA-49, TCGA-73, etc. are LUAD
    # TCGA-BH, TCGA-A2, TCGA-E2, TCGA-A8, etc. are BRCA
    # Use a broader approach: just group by cohort prefix
    patient_cohort = pd.Series(ids).str[:7]

    cancer_specific_results = []
    # Pick the two largest cohorts that have both classes
    for cohort_prefix, cohort_name in [("TCGA-BH", "BRCA (breast)"),
                                        ("TCGA-CV", "HNSC (head/neck)"),
                                        ("TCGA-AA", "COAD (colon)"),
                                        ("TCGA-HT", "LGG (brain)")]:
        mask = patient_cohort.values == cohort_prefix
        if mask.sum() < 30:
            continue

        X_c = X_bert[mask]
        y_c = y[mask]
        n_pos_c = int(y_c.sum())
        n_neg_c = int((y_c == 0).sum())

        if n_pos_c < 5 or n_neg_c < 5:
            print(f"  {cohort_name}: insufficient class balance, skipping")
            continue

        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
            X_c, y_c, test_size=0.2, stratify=y_c, random_state=RANDOM_STATE
        )

        model_c = LogisticRegression(class_weight="balanced", C=1.0,
                                     max_iter=1000, random_state=RANDOM_STATE)
        model_c.fit(Xc_tr, yc_tr)
        yc_prob = model_c.predict_proba(Xc_te)[:, 1]
        auc_c = roc_auc_score(yc_te, yc_prob)

        cancer_specific_results.append({
            "Cancer_Type": cohort_name, "Cohort": cohort_prefix,
            "N_Patients": int(mask.sum()), "N_Tgt": n_pos_c,
            "AUC": round(auc_c, 4),
        })
        print(f"  {cohort_name:<25} n={mask.sum():>4}  tgt={n_pos_c:>4}  AUC={auc_c:.4f}")

        with open(os.path.join(BERT_DIR, f"model_{cohort_prefix.replace('-','_')}.pkl"), "wb") as f:
            pickle.dump(model_c, f)

    # ════════════════════════════════════════════════════════════════════
    # STEP 7: Inference function definition
    # ════════════════════════════════════════════════════════════════════
    section("STEP 7: Inference function")

    # Save an inference-ready bundle
    inference_bundle = {
        "bert_model_name": "see bert_embeddings.py for model used",
        "classifier": best_bert_model,
        "gene_models": {gene: os.path.join(per_gene_dir, f"{gene}_model.pkl")
                        for gene in TOP_GENES},
        "top_genes": TOP_GENES,
        "embedding_dim": X_bert.shape[1],
    }
    bundle_path = os.path.join(BERT_DIR, "inference_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(inference_bundle, f)
    print(f"  Saved: {bundle_path}")

    # Write the inference helper script
    inference_code = '''"""
Inference Helper -- Predict from a new pathology report
========================================================
Usage:
    from bert_inference import predict_new_report
    result = predict_new_report("Patient presents with invasive ductal carcinoma...")
"""
import os, re, pickle, numpy as np

BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s.\\-]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def get_embedding(text, model_name=BERT_MODEL_NAME):
    """Get ClinicalBERT embedding for a single report (GPU-aware)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    encoded = tokenizer(text, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    emb = outputs.last_hidden_state.float()
    pooled = ((emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
    return pooled

def predict_new_report(report_text: str) -> dict:
    """Predict targetable mutation status from a pathology report."""
    clean = clean_text(report_text)
    embedding = get_embedding(clean)

    # Load classifier
    model_dir = "bert_models"
    with open(os.path.join(model_dir, "best_bert_model.pkl"), "rb") as f:
        clf = pickle.load(f)

    prob = clf.predict_proba(embedding)[0, 1]
    pred_class = "Targetable_Mutation" if prob >= 0.5 else "Non_Targetable"

    # Per-gene predictions
    gene_dir = os.path.join(model_dir, "per_gene_models")
    likely_genes = []
    for fname in os.listdir(gene_dir):
        if fname.endswith("_model.pkl"):
            gene = fname.replace("_model.pkl", "")
            with open(os.path.join(gene_dir, fname), "rb") as f:
                gm = pickle.load(f)
            gp = gm.predict_proba(embedding)[0, 1]
            if gp >= 0.3:
                likely_genes.append((gene, round(gp, 3)))
    likely_genes.sort(key=lambda x: -x[1])

    follow_up = {
        "Targetable_Mutation": "URGENT: Refer to oncologist -- targeted therapy indicated",
        "Non_Targetable": "MONITOR: Standard-of-care; consider clinical trial",
    }

    confidence_note = ""
    if 0.4 <= prob <= 0.6:
        confidence_note = " | LOW CONFIDENCE -- manual review recommended"

    return {
        "predicted_class": pred_class,
        "confidence": round(prob, 4),
        "top_likely_genes": likely_genes[:5],
        "follow_up": follow_up[pred_class] + confidence_note,
        "model_used": f"ClinicalBERT + Classifier",
    }

if __name__ == "__main__":
    import torch
    print(f"  Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    demo = "Invasive ductal carcinoma, grade 3, ER positive, PR negative, HER2 positive."
    result = predict_new_report(demo)
    for k, v in result.items():
        print(f"  {k}: {v}")
'''
    inf_path = os.path.join(BERT_DIR, "bert_inference.py")
    with open(inf_path, "w") as f:
        f.write(inference_code)
    print(f"  Saved: {inf_path}")
    print(f"  Usage: from bert_models.bert_inference import predict_new_report")

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start
    section("PHASE 6 COMPLETE")

    print(f"""
  BERT embedding dim        : {X_bert.shape[1]}
  Best BERT model           : {best_bert_name} (AUC = {best_bert_auc:.4f})
  Best TF-IDF model         : LR (AUC = {tfidf_solo_auc:.4f})
  Ensemble AUC              : {ensemble_auc:.4f}
  Per-gene models trained   : {len(gene_results)}
  Cancer-specific models    : {len(cancer_specific_results)}

  Artifacts saved:
    bert_models/best_bert_model.pkl
    bert_models/per_gene_models/*.pkl
    bert_models/bert_inference.py
    output/ensemble_model_comparison.tsv
    output/per_gene_auc.tsv
    output/roc_bert_ensemble.png

  Time elapsed: {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
