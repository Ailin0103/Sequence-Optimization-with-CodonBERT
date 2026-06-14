"""
k-mer baseline for Cambray 2018 expression dataset.

This script tests whether simple local sequence motifs can predict:
1. protein_mean
2. gs.utrCdsStructureMFE

Compared with CodonBERT/RNABERT embeddings, this is a lightweight
traditional ML baseline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =====================
# Config
# =====================

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
)

SEQ_COL = "gs.sequence"
EXPR_TARGET_COL = "protein_mean"
MFE_TARGET_COL = "gs.utrCdsStructureMFE"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = 2

# Use 10000 first. Increase later if fast.
SAMPLE_SIZE = 10000

# Try 3, 4, 5. 4 is often a good first choice.
K_VALUES = [3, 4, 5]
K_VALUES = [8,9,10,11]
# Feature type: "tfidf" or "count"
VECTORIZER_TYPE = "tfidf"


# =====================
# Helpers
# =====================

def clean_sequence(seq):
    """Clean sequence and keep only A/T/C/G."""
    seq = str(seq).upper().strip()
    seq = seq.replace("U", "T")

    # Drop sequences with weird bases for now
    if not all(c in "ATCG" for c in seq):
        return None

    return seq


def sequence_to_kmer_text(seq, k):
    """
    Convert sequence to overlapping k-mer text.

    Example:
    ATGCG, k=3 -> ATG TGC GCG
    """
    seq = clean_sequence(seq)

    if seq is None:
        return None

    if len(seq) < k:
        return None

    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    return " ".join(kmers)


def build_targets(df):
    """Create protein_mean from four log protein replicates."""
    rep_cols = [
        "log.prot.repOne",
        "log.prot.repTwo",
        "log.prot.repThree",
        "log.prot.repFour",
    ]

    df[EXPR_TARGET_COL] = df[rep_cols].mean(axis=1)
    return df


def evaluate_regression(name, model, X_train, X_test, y_train, y_test):
    """Fit model and print metrics."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    spearman = spearmanr(y_test, pred).correlation

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")

    return {
        "model": name,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }


def get_vectorizer(k):
    """Return k-mer vectorizer."""
    if VECTORIZER_TYPE == "tfidf":
        return TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )

    if VECTORIZER_TYPE == "count":
        return CountVectorizer(
            analyzer="word",
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )

    raise ValueError(f"Unknown VECTORIZER_TYPE: {VECTORIZER_TYPE}")


def run_for_target(df, k, target_col):
    """Run k-mer models for one target."""
    print("\n" + "#" * 80)
    print(f"k = {k}, target = {target_col}")
    print("#" * 80)

    df_k = df.copy()
    df_k["kmer_text"] = df_k[SEQ_COL].apply(lambda s: sequence_to_kmer_text(s, k))
    df_k = df_k.dropna(subset=["kmer_text", target_col])

    print("Data shape:", df_k.shape)

    X_text = df_k["kmer_text"].values
    y = df_k[target_col].astype(float).values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    vectorizer = get_vectorizer(k)

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Feature shape:", X_train.shape)

    results = []

    # Ridge works well with sparse high-dimensional features
    ridge = Ridge(alpha=1.0)
    
    results.append(
        evaluate_regression(
            f"k{k}-{VECTORIZER_TYPE} + Ridge alpha={1} -> {target_col}",
            ridge,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    # ===== analyze coefficients =====

    if k == 8 and target_col == "gs.utrCdsStructureMFE":

        coef_df = pd.DataFrame({

            "kmer": vectorizer.get_feature_names_out(),

            "coef": ridge.coef_

        })

        coef_df["abs_coef"] = coef_df["coef"].abs()

        print("\nTOP 20 POSITIVE KMERS")

        print(

            coef_df.sort_values(

                "coef",

                ascending=False

            )[["kmer", "coef"]].head(20)

        )

        print("\nTOP 20 NEGATIVE KMERS")

        print(

            coef_df.sort_values(

                "coef",

                ascending=True

            )[["kmer", "coef"]].head(20)
        )
        coef_df.sort_values(
        "abs_coef",
        ascending=False
        ).to_csv(
        "k8_mfe__tailcoefficients.csv",
        index=False
        )
    # Random Forest can be slow on sparse high-dimensional features.
    # Try a light version only.
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    results.append(
        evaluate_regression(
            f"k{k}-{VECTORIZER_TYPE} + RF -> {target_col}",
            rf,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    return results
    

# =====================
# Main
# =====================

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print("Raw shape:", df.shape)

    df = build_targets(df)

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

    print("\nTarget summaries:")
    print(df[[EXPR_TARGET_COL, MFE_TARGET_COL]].describe())

    all_results = []

    for k in K_VALUES:
        all_results.extend(run_for_target(df, k, EXPR_TARGET_COL))
        all_results.extend(run_for_target(df, k, MFE_TARGET_COL))
        

    results_df = pd.DataFrame(all_results)

    output_path = (
        Path(__file__).parent.parent
        / "data"
        / "kmer_extended_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nSaved results to:", output_path)
    print(results_df)


if __name__ == "__main__":
    main()
