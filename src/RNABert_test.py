"""
RNABERT baseline for Cambray 2018 expression dataset.

Goal:
1. Use RNABERT nucleotide-level embeddings.
2. Predict protein_mean.
3. Predict gs.utrCdsStructureMFE.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from multimolecule import AutoTokenizer, AutoModel


# =====================
# Config
# =====================

MODEL_NAME = "multimolecule/rnabert"

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
)

SEQ_COL = "gs.sequence"
EXPR_TARGET_COL = "protein_mean"
MFE_TARGET_COL = "gs.utrCdsStructureMFE"

MODEL_BATCH_SIZE = 16
MAX_LEN = 128
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Set sample size for fast testing.
# Use None for full data.
SAMPLE_SIZE = 10000


# =====================
# Helpers
# =====================

def dna_to_rna(seq):
    """Convert DNA sequence to RNA sequence."""
    return str(seq).upper().replace("T", "U")


def get_device():
    """Use Apple GPU if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_rnabert_embeddings(sequences, tokenizer, model, device):
    """
    Compute mean-pooled RNABERT embeddings.

    RNABERT expects RNA nucleotide strings, not codon-spaced strings.
    """
    all_embeddings = []

    for start in tqdm(
        range(0, len(sequences), MODEL_BATCH_SIZE),
        desc="RNABERT embedding",
    ):
        batch_seqs = sequences[start:start + MODEL_BATCH_SIZE]

        batch_seqs = [dna_to_rna(seq) for seq in batch_seqs]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(pooled.detach().cpu().numpy())

    return np.vstack(all_embeddings)


def evaluate_regression(name, X, y):
    """Train/test Ridge regression and print metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    spearman = spearmanr(y_test, pred).correlation

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")

    return {
        "model": name,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }


# =====================
# Main
# =====================

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Raw shape:", df.shape)

    # Build protein_mean from four protein expression replicates
    rep_cols = [
        "log.prot.repOne",
        "log.prot.repTwo",
        "log.prot.repThree",
        "log.prot.repFour",
    ]

    df[EXPR_TARGET_COL] = df[rep_cols].mean(axis=1)

    keep_cols = [
        SEQ_COL,
        EXPR_TARGET_COL,
        MFE_TARGET_COL,
    ]

    df = df[keep_cols].dropna()

    print("After dropping NA:", df.shape)

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"After sampling {SAMPLE_SIZE}:", df.shape)

    sequences = df[SEQ_COL].astype(str).tolist()

    y_expr = df[EXPR_TARGET_COL].astype(float).values
    y_mfe = df[MFE_TARGET_COL].astype(float).values

    print("\nTarget summaries:")
    print(df[[EXPR_TARGET_COL, MFE_TARGET_COL]].describe())

    device = get_device()
    print("\nUsing device:", device)

    print("\nLoading RNABERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("\nComputing RNABERT embeddings...")
    X = compute_rnabert_embeddings(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    print("Embedding shape:", X.shape)

    results = []

    results.append(
        evaluate_regression(
            "RNABERT + Ridge -> protein_mean",
            X,
            y_expr,
        )
    )

    results.append(
        evaluate_regression(
            "RNABERT + Ridge -> gs.utrCdsStructureMFE",
            X,
            y_mfe,
        )
    )

    results_df = pd.DataFrame(results)

    output_path = (
        Path(__file__).parent.parent
        / "data"
        / "rnabert_baseline_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nSaved results to:", output_path)
    print(results_df)


if __name__ == "__main__":
    main()