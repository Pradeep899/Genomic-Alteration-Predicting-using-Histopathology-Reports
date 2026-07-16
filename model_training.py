"""
Model Training & Evaluation Pipeline (Phase 4)
================================================
Trains Logistic Regression, Random Forest, and SGD Classifier
on TF-IDF features for binary classification of targetable mutations.

All models natively support sparse matrices -- no dense conversion needed.

Input:  output/X_tfidf.npz, y_binary.npy, patient_ids.npy, tfidf_vectorizer.pkl
Output: output/best_model.pkl, model_results.tsv, roc_curves.png,
        confusion_matrices.png, feature_importance.tsv
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Force unbuffered output so we see progress in real-time
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
X_PATH = os.path.join(OUTPUT_DIR, "X_tfidf.npz")
Y_PATH = os.path.join(OUTPUT_DIR, "y_binary.npy")
IDS_PATH = os.path.join(OUTPUT_DIR, "patient_ids.npy")
VEC_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")

RANDOM_STATE = 42
N_CV_FOLDS = 5


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_class_dist(y, label=""):
    total = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"  {label}")
    print(f"    Total: {total:,}  |  Positive (Targetable): {pos:,} ({pos/total*100:.1f}%)  "
          f"|  Negative (Other): {neg:,} ({neg/total*100:.1f}%)")


def main():
    start = time.time()

    # ================================================================
    # STEP 0 -- Load data
    # ================================================================
    section("STEP 0: Loading artifacts")

    X = sparse.load_npz(X_PATH)
    y = np.load(Y_PATH)
    patient_ids = np.load(IDS_PATH, allow_pickle=True)

    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    feature_names = np.array(vectorizer.get_feature_names_out())

    print(f"  X shape       : {X.shape}")
    print(f"  y shape       : {y.shape}")
    print(f"  Features      : {len(feature_names):,}")
    print_class_dist(y, "Full dataset:")

    # ================================================================
    # STEP 1 -- Train/test split
    # ================================================================
    section("STEP 1: Train/test split (80/20, stratified)")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print_class_dist(y_train, "Train set:")
    print_class_dist(y_test, "Test set:")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"\n  scale_pos_weight = {neg_count}/{pos_count} = {scale_pos_weight:.4f}")

    # ================================================================
    # STEP 2 -- Define & train models
    # ================================================================
    section("STEP 2: Training models")

    models = {}

    # --- Model A: Logistic Regression (L2, LBFGS) ---
    print("\n  [A] Logistic Regression (LBFGS, L2) ...")
    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    print(f"      Trained in {time.time()-t0:.1f}s")

    # --- Model B: Random Forest ---
    print("  [B] Random Forest (300 trees) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print(f"      Trained in {time.time()-t0:.1f}s")

    # --- Model C: SGD Classifier (modified Huber loss, elastic-net) ---
    # SGDClassifier with log_loss + calibration gives us a fast, regularized
    # linear model with elastic-net penalty -- genuinely different from LR.
    # Natively handles sparse matrices, no dense conversion needed.
    print("  [C] SGD Classifier (log_loss, elastic-net, calibrated) ...")
    t0 = time.time()
    sgd_raw = SGDClassifier(
        loss="modified_huber",     # outputs probabilities directly
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.15,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    sgd_raw.fit(X_train, y_train)
    models["SGD Classifier"] = sgd_raw
    print(f"      Trained in {time.time()-t0:.1f}s")

    print(f"\n  All 3 models trained in {time.time()-start:.1f}s total")

    # ================================================================
    # STEP 3 -- Evaluation
    # ================================================================
    section("STEP 3: Model evaluation")

    results = []
    roc_data = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")

        # Predictions & probabilities
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.decision_function(X_train)
            y_test_prob = model.decision_function(X_test)

        y_pred = model.predict(X_test)

        # AUC
        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc = roc_auc_score(y_test, y_test_prob)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_data[name] = (fpr, tpr, test_auc)

        # Cross-validation AUC
        print(f"    Running {N_CV_FOLDS}-fold stratified CV ...")
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Per-class metrics
        f1_tgt = f1_score(y_test, y_pred, pos_label=1)
        prec_tgt = precision_score(y_test, y_pred, pos_label=1)
        recall_tgt = recall_score(y_test, y_pred, pos_label=1)

        print(f"    Train AUC        : {train_auc:.4f}")
        print(f"    Test AUC         : {test_auc:.4f}")
        print(f"    CV AUC           : {cv_mean:.4f} +/- {cv_std:.4f}")
        print(f"    F1 (Targetable)  : {f1_tgt:.4f}")
        print(f"    Precision        : {prec_tgt:.4f}")
        print(f"    Recall           : {recall_tgt:.4f}")

        print(f"\n    Classification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=["Non-Targetable", "Targetable"],
            digits=4,
        )
        for line in report.split("\n"):
            print(f"      {line}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"    Confusion Matrix:")
        print(f"      Predicted:   Non-Tgt   Tgt")
        print(f"      Non-Tgt    {cm[0,0]:>7,}  {cm[0,1]:>5,}")
        print(f"      Tgt        {cm[1,0]:>7,}  {cm[1,1]:>5,}")

        results.append({
            "Model": name,
            "Train_AUC": round(train_auc, 4),
            "Test_AUC": round(test_auc, 4),
            "CV_AUC_mean": round(cv_mean, 4),
            "CV_AUC_std": round(cv_std, 4),
            "F1_Targetable": round(f1_tgt, 4),
            "Precision": round(prec_tgt, 4),
            "Recall": round(recall_tgt, 4),
            "confusion_matrix": cm,
        })

    # ================================================================
    # STEP 4 -- Comparison table
    # ================================================================
    section("STEP 4: Model comparison")

    results_df = pd.DataFrame(results)

    print(f"\n  {'Model':<25} {'Train AUC':>10} {'Test AUC':>10} "
          f"{'CV AUC':>16} {'F1-Tgt':>8} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*16} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in results_df.iterrows():
        cv_str = f"{row['CV_AUC_mean']:.4f}+/-{row['CV_AUC_std']:.4f}"
        print(f"  {row['Model']:<25} {row['Train_AUC']:>10.4f} {row['Test_AUC']:>10.4f} "
              f"{cv_str:>16} {row['F1_Targetable']:>8.4f} {row['Precision']:>8.4f} {row['Recall']:>8.4f}")

    # Best model by test AUC
    best_idx = results_df["Test_AUC"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_auc = results_df.loc[best_idx, "Test_AUC"]
    best_model = models[best_name]

    print(f"\n  >>> BEST MODEL: {best_name} (Test AUC = {best_auc:.4f})")

    # Overfitting check
    train_auc_best = results_df.loc[best_idx, "Train_AUC"]
    gap = train_auc_best - best_auc
    if gap > 0.10:
        print(f"  [WARN] Overfitting detected: Train-Test AUC gap = {gap:.4f}")
    elif gap > 0.05:
        print(f"  [NOTE] Slight overfitting: Train-Test AUC gap = {gap:.4f}")
    else:
        print(f"  [OK]   Good generalization: Train-Test AUC gap = {gap:.4f}")

    # Why this model wins
    runner_up_idx = results_df.loc[results_df.index != best_idx, "Test_AUC"].idxmax()
    runner_up = results_df.loc[runner_up_idx]
    print(f"\n  WHY: {best_name} achieves {best_auc:.4f} AUC vs runner-up "
          f"{runner_up['Model']} at {runner_up['Test_AUC']:.4f}.")
    if gap < 0.05:
        print(f"         It also generalizes well (train-test gap = {gap:.4f}).")

    # ================================================================
    # STEP 5 -- Feature importance
    # ================================================================
    section("STEP 5: Feature importance (best model)")

    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.tsv")

    # For linear models (LR, SGD), use coefficients
    if hasattr(best_model, "coef_"):
        coefs = best_model.coef_[0]

        top_pos_idx = np.argsort(coefs)[::-1][:30]
        top_neg_idx = np.argsort(coefs)[:30]

        print(f"\n  Top 30 words predicting TARGETABLE mutation:")
        print(f"    {'Rank':<6} {'Feature':<30} {'Coefficient':>12}")
        print(f"    {'-'*6} {'-'*30} {'-'*12}")
        fi_rows = []
        for rank, idx in enumerate(top_pos_idx, 1):
            print(f"    {rank:<6} {feature_names[idx]:<30} {coefs[idx]:>12.4f}")
            fi_rows.append({"direction": "Targetable", "rank": rank,
                            "feature": feature_names[idx], "coefficient": round(coefs[idx], 6)})

        print(f"\n  Top 30 words predicting NON-TARGETABLE:")
        print(f"    {'Rank':<6} {'Feature':<30} {'Coefficient':>12}")
        print(f"    {'-'*6} {'-'*30} {'-'*12}")
        for rank, idx in enumerate(top_neg_idx, 1):
            print(f"    {rank:<6} {feature_names[idx]:<30} {coefs[idx]:>12.4f}")
            fi_rows.append({"direction": "Non-Targetable", "rank": rank,
                            "feature": feature_names[idx], "coefficient": round(coefs[idx], 6)})

        fi_df = pd.DataFrame(fi_rows)

    else:
        # Tree-based: feature_importances_
        importances = best_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:30]

        print(f"\n  Top 30 features by importance:")
        print(f"    {'Rank':<6} {'Feature':<30} {'Importance':>12}")
        print(f"    {'-'*6} {'-'*30} {'-'*12}")
        fi_rows = []
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:<6} {feature_names[idx]:<30} {importances[idx]:>12.6f}")
            fi_rows.append({"rank": rank, "feature": feature_names[idx],
                            "importance": round(importances[idx], 8)})

        fi_df = pd.DataFrame(fi_rows)

    fi_df.to_csv(fi_path, sep="\t", index=False)
    print(f"\n  Saved: {fi_path}")

    # ================================================================
    # STEP 6 -- ROC curve plot
    # ================================================================
    section("STEP 6: Plotting ROC curves")

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#4361ee", "#f72585", "#4cc9f0"]
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        marker = " ***" if name == best_name else ""
        ax.plot(fpr, tpr, color=colors[i], lw=2.2,
                label=f"{name} (AUC = {auc_val:.4f}){marker}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves -- Targetable Mutation Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {roc_path}")

    # ================================================================
    # STEP 7 -- Confusion matrix plots
    # ================================================================
    section("STEP 7: Plotting confusion matrices")

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    class_labels = ["Non-Targetable", "Targetable"]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{res['Model']}\nAUC = {res['Test_AUC']:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color=color)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    fig.suptitle("Confusion Matrices -- Binary Classification", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {cm_path}")

    # ================================================================
    # STEP 8 -- Save best model & results table
    # ================================================================
    section("STEP 8: Saving best model & results")

    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Saved: {model_path} ({os.path.getsize(model_path)/1e6:.1f} MB)")
    print(f"         Model: {best_name}")

    results_table = results_df.drop(columns=["confusion_matrix"])
    results_table["CV_AUC"] = results_table.apply(
        lambda r: f"{r['CV_AUC_mean']:.4f}+/-{r['CV_AUC_std']:.4f}", axis=1
    )
    results_path = os.path.join(OUTPUT_DIR, "model_results.tsv")
    results_table.to_csv(results_path, sep="\t", index=False)
    print(f"  Saved: {results_path}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start
    section("TRAINING PIPELINE COMPLETE")

    print(f"""
  Dataset          : {X.shape[0]:,} patients x {X.shape[1]:,} features
  Train / Test     : {X_train.shape[0]:,} / {X_test.shape[0]:,}
  Models trained   : {len(models)}
  Best model       : {best_name}
  Best Test AUC    : {best_auc:.4f}
  CV AUC           : {results_df.loc[best_idx, 'CV_AUC_mean']:.4f} +/- {results_df.loc[best_idx, 'CV_AUC_std']:.4f}
  
  Artifacts saved to: {OUTPUT_DIR}/
    - best_model.pkl
    - model_results.tsv
    - roc_curves.png
    - confusion_matrices.png
    - feature_importance.tsv

  Time elapsed: {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
