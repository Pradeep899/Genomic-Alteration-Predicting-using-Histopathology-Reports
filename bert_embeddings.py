"""
BERT Embedding Generation (Phase 6a)
=====================================
Encodes pathology reports using Bio_ClinicalBERT (768-dim) into dense
embeddings for downstream ML. Auto-detects GPU and uses it if available.

Output: output/X_bert.npy
"""

import os, sys, time, warnings, numpy as np, pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "output"
REPORTS_FILE = os.path.join(OUTPUT_DIR, "clean_reports.tsv")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "X_bert.npy")

# Always use ClinicalBERT for best accuracy
PRIMARY_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FALLBACK_MODEL = "all-MiniLM-L6-v2"

# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 if DEVICE == "cuda" else 16
MAX_LENGTH = 512


def encode_with_transformers(texts, model_name, batch_size):
    """Encode texts using HuggingFace transformers with mean pooling."""
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # Use half precision on GPU for speed
    if DEVICE == "cuda":
        model = model.half()
        print(f"  Using FP16 on GPU for faster inference")

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=n_batches,
                      desc="  Encoding", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state.float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings)


def encode_with_sentence_transformers(texts, model_name, batch_size):
    """Encode texts using sentence-transformers (fallback)."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def main():
    start = time.time()

    print("=" * 70)
    print("  BERT EMBEDDING GENERATION")
    print("=" * 70)
    print(f"  Device           : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Max token length : {MAX_LENGTH}")
    print(f"  Primary model    : {PRIMARY_MODEL}")

    if DEVICE == "cpu":
        print(f"\n  [WARNING] Running on CPU -- ClinicalBERT will take ~3-6 hours.")
        print(f"  For faster results, run on a GPU-enabled machine.")
        est_minutes = 240
    else:
        est_minutes = 10
    print(f"  Estimated time   : ~{est_minutes} minutes")

    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_FILE):
        existing = np.load(EMBEDDINGS_FILE)
        print(f"\n  [SKIP] Embeddings already exist: {EMBEDDINGS_FILE}")
        print(f"         Shape: {existing.shape}")
        print(f"         To regenerate, delete the file and re-run.")
        return

    # Load reports
    print("\n  Loading clean reports ...")
    df = pd.read_csv(REPORTS_FILE, sep="\t")
    texts = df["clean_text"].fillna("").tolist()
    print(f"  Reports loaded: {len(texts):,}")

    # Truncate to MAX_LENGTH words
    max_words = MAX_LENGTH - 20
    texts = [" ".join(t.split()[:max_words]) for t in texts]

    # Always try ClinicalBERT first
    try:
        print(f"\n  Using primary model: {PRIMARY_MODEL}")
        embeddings = encode_with_transformers(texts, PRIMARY_MODEL, BATCH_SIZE)
        model_used = PRIMARY_MODEL
    except Exception as e:
        print(f"\n  [WARN] ClinicalBERT failed: {e}")
        print(f"  Falling back to: {FALLBACK_MODEL}")
        try:
            embeddings = encode_with_sentence_transformers(
                texts, FALLBACK_MODEL, BATCH_SIZE
            )
            model_used = FALLBACK_MODEL
        except Exception as e2:
            print(f"  [ERROR] Fallback also failed: {e2}")
            sys.exit(1)

    # Save
    np.save(EMBEDDINGS_FILE, embeddings)
    elapsed = time.time() - start

    print(f"\n  {'=' * 50}")
    print(f"  Embeddings saved : {EMBEDDINGS_FILE}")
    print(f"  Shape            : {embeddings.shape}")
    print(f"  Model used       : {model_used}")
    print(f"  Embedding dim    : {embeddings.shape[1]}")
    print(f"  File size        : {os.path.getsize(EMBEDDINGS_FILE)/1e6:.1f} MB")
    print(f"  Time elapsed     : {elapsed/60:.1f} minutes")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
