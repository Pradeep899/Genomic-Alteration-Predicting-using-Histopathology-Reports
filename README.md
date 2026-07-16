Genomic Alteration Prediction using Histopathology Reports
============================================================

Predicting targetable cancer mutations from pathology report text alone -
no genomic sequencing required.


WHAT THIS PROJECT DOES
------------------------------------------------------------
Genomic sequencing tells doctors which cancer mutations a patient has, and
whether any of those mutations can be treated with an existing FDA-approved
targeted drug. The problem is sequencing takes time and costs money.

Every patient already gets something faster and cheaper though: a
pathology report, written by a doctor right after their biopsy is examined.

This project asks a simple question - can we predict a patient's likely
mutation status just from that report text, before sequencing results even
come back? If it works even reasonably well, it could help hospitals
prioritize which patients most urgently need confirmatory genomic testing.

This is a pan-cancer project, meaning it works across many cancer types at
once (breast, lung, brain, colon, etc.), not just one. It never looks at
DNA data directly during prediction - only the doctor's written words.


HOW IT WORKS (PIPELINE OVERVIEW)
------------------------------------------------------------
The project runs in 6 phases. Each phase saves its own output, so later
phases never have to re-touch the large raw files.

  Phase 1 - data_alignment.py
    Reads a 3.5 GB mutation file (MC3 MAF) in small chunks so it never
    overloads memory, then matches it up with pathology report text by
    patient ID.

  Phase 2 - mutation_labeling.py
    Looks at each patient's mutated genes and labels them as:
      - Targetable_Mutation      (has a mutation with an approved drug)
      - Non_Targetable_Mutation  (has mutations, but none druggable)
      - No_Mutation              (no meaningful mutation found)
    Uses a curated list of 59 clinically actionable genes.

  Phase 3 - feature_engineering.py
    Cleans the report text (expands medical abbreviations, strips
    boilerplate/signatures) and converts it into TF-IDF features -
    a 25,000-dimension numeric representation of the words that matter.

  Phase 4 - model_training.py
    Trains Logistic Regression, Random Forest, and SGD classifiers on the
    TF-IDF features and picks the best one by AUC.

  Phase 5 - clinical_analysis.py
    Turns the model's predictions into something a clinician could
    actually use - per-patient confidence scores, drug recommendations,
    and cancer-type-specific performance breakdowns.

  Phase 6 - bert_embeddings.py + bert_training.py
    Re-encodes every report using Bio_ClinicalBERT, a deep learning model
    already trained on clinical text, and retrains the same models on
    those embeddings. Then combines TF-IDF and BERT predictions into an
    ensemble for the final, strongest result.


RESULTS
------------------------------------------------------------
  Dataset size                 : 8,796 patients (TCGA, pan-cancer)
  Best TF-IDF model             : Logistic Regression, 0.7329 AUC
  Best ensemble (TF-IDF + BERT) : 0.7382 AUC
  Best single-gene result       : IDH1, 0.9187 AUC (drug: Ivosidenib)
  Targetable mutation rate      : 66.8% of patients in this cohort

Some genes are much easier to predict than others. IDH1, KRAS, and BRAF
mutations tend to show up in very distinctive report language (specific
cancer subtypes, grading terms), so those genes score highest. Others,
like BRCA2, are noisier and harder to call from text alone.


DATASET
------------------------------------------------------------
Two files are needed, both from the GDC Data Portal (TCGA):

  mc3.v0.2.8.PUBLIC.maf   (~3.5 GB)  - TCGA pan-cancer mutation calls
  TCGA_Reports.csv        (~30 MB)  - pathology report text per patient

These are not included in this repo due to size. Download them yourself
from https://portal.gdc.cancer.gov/ and place both files in the project
root folder before running anything.


HOW TO RUN IT
------------------------------------------------------------
1. Install dependencies:
     pip install -r requirements.txt

2. Place mc3.v0.2.8.PUBLIC.maf and TCGA_Reports.csv in the project root.

3. Run the scripts in this exact order (each one depends on the last):
     python data_alignment.py
     python mutation_labeling.py
     python feature_engineering.py
     python model_training.py
     python clinical_analysis.py

4. Optional but recommended - the deep learning phase (works on CPU too,
   just much slower):
     python bert_embeddings.py
     python bert_training.py

All scripts auto-detect GPU and fall back to CPU automatically. Expect
the whole pipeline (without BERT) to finish in under 15 minutes. The BERT
phase takes about 10 minutes on GPU, or a few hours on CPU alone.


WHAT'S IN THE output/ FOLDER AFTER RUNNING
------------------------------------------------------------
  mutations_processed.tsv       - cleaned mutation records
  patient_mutation_labels.tsv   - each patient's mutation class
  X_tfidf.npz / clean_reports.tsv - text features
  best_model.pkl                - the winning TF-IDF model
  roc_curves.png / confusion_matrices.png - model performance plots
  final_patient_report.tsv      - per-patient predictions + drug mapping
  summary_report.txt            - plain-text overview of everything
  X_bert.npy                    - BERT embeddings (if Phase 6 was run)
  roc_bert_ensemble.png         - TF-IDF vs BERT vs Ensemble comparison


TECH STACK
------------------------------------------------------------
  pandas, numpy, scipy       - data wrangling
  scikit-learn               - Logistic Regression, Random Forest, SGD
  PyTorch + transformers     - Bio_ClinicalBERT embeddings
  matplotlib                 - plots


HONEST LIMITATIONS
------------------------------------------------------------
This is a research/portfolio project, not a clinical tool. An AUC around
0.73-0.74 is a decent signal that report text correlates with mutation
status, but it is far from diagnostic-grade accuracy. Performance also
varies a lot by gene and cancer type - treat per-gene numbers as a guide,
not a guarantee, and always confirm with actual genomic sequencing before
making any treatment decision.


CITATION
------------------------------------------------------------
If you use this project or its approach, please credit the original data
sources:
  - TCGA MC3 Mutation Data: Ellrott et al., Cell Systems (2018)
  - Bio_ClinicalBERT: Alsentzer et al., ACL Clinical NLP Workshop (2019)
