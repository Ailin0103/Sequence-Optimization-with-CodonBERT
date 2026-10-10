"""
Full-data 7-mer TF-IDF + Ridge baseline for the Cambray 2018
protein-expression prediction and favorable-expression screening task.

Goal
----
Predict experimental protein expression from nucleotide sequence using:

    overlapping 7-mer TF-IDF features
    +
    Ridge regression

This script is aligned with the final CodonBERT experiment:

    input:
        gs.sequence

    target:
        clean.lin.prot.mean

    dataset:
        all valid Cambray sequences

    random seed:
        42

    split:
        80% train
        10% validation
        10% test

The Ridge model predicts protein expression directly on the original PNI scale.
The target is not standardized.

In addition to regression performance, this script evaluates the k-mer model
as a screening tool for sequences in a fitness-favorable expression range.

Based on the Cambray dataset, average cellular fitness is near its maximum
for intermediate protein production. We therefore define experimentally
measured PNI 40-60 as the favorable expression range.

The regression model itself is not tuned using the test set. Instead:

1. Ridge is trained only on the training set.
2. The validation set is used to select a predicted-expression interval that:
   - achieves at least the specified minimum recall for truly favorable
     sequences, and
   - maximizes enrichment among selected candidates.
3. The selected prediction interval is frozen.
4. Final screening performance is evaluated once on the held-out test set.

This allows direct comparison with the CodonBERT screening workflow.

Main outputs:
    - metrics.csv
    - validation_predictions.csv
    - test_predictions.csv
    - validation_favorable_interval_scan.csv
    - favorable_screening_summary.json
"""


from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Config
# ============================================================

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "kmer7_baseline"
)

SEQ_COL = "gs.sequence"
TARGET_COL = "clean.lin.prot.mean"

RANDOM_STATE = 42

K = 7
RIDGE_ALPHA = 1.0


# ============================================================
# Favorable-expression screening definition
# ============================================================

TRUE_RANGE_LOW = 40.0
TRUE_RANGE_HIGH = 60.0

MIN_RECALL = 0.80

INTERVAL_STEP = 1.0


# ============================================================
# Sequence preprocessing
# ============================================================

def clean_sequence(seq):
    """
    Clean sequence using the same biological validity rules
    used for the CodonBERT experiment.

    Output remains in the DNA alphabet A/T/G/C because the
    k-mer model does not require RNA conversion.
    """

    if pd.isna(seq):
        return None

    seq = str(seq).upper().strip()

    # Normalize RNA -> DNA if needed
    seq = seq.replace("U", "T")

    if any(
        base not in {"A", "T", "G", "C"}
        for base in seq
    ):
        return None

    # Match CodonBERT frame compatibility
    if len(seq) % 3 != 0:
        return None

    return seq


def sequence_to_kmer_text(
    seq,
    k=K,
):
    """
    Convert sequence into overlapping k-mer tokens.

    Example
    -------
    ATGCG, k=3

    becomes:

    ATG TGC GCG
    """

    seq = clean_sequence(seq)

    if seq is None:
        return None

    if len(seq) < k:
        return None

    kmers = [
        seq[i:i + k]
        for i in range(
            len(seq) - k + 1
        )
    ]

    return " ".join(kmers)


# ============================================================
# Load full Cambray dataset
# ============================================================

def load_cambray_data():

    print("\n==============================")
    print("LOADING DATA")
    print("==============================")

    print(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "\nOriginal rows:",
        len(df),
    )

    df = df[
        [
            SEQ_COL,
            TARGET_COL,
        ]
    ].copy()

    # Ensure numeric target first
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            SEQ_COL,
            TARGET_COL,
        ]
    )

    df["clean_sequence"] = (
        df[SEQ_COL]
        .apply(clean_sequence)
    )

    before = len(df)

    df = (
        df
        .dropna(
            subset=[
                "clean_sequence"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\nValid rows:",
        len(df),
    )

    print(
        "Skipped rows:",
        before - len(df),
    )

    lengths = (
        df["clean_sequence"]
        .str.len()
    )

    print(
        "\nSequence length summary:"
    )

    print(
        lengths.describe()
    )

    print(
        "\nMost common sequence lengths:"
    )

    print(
        lengths
        .value_counts()
        .head(10)
    )

    print(
        "\nTarget summary:"
    )

    print(
        df[TARGET_COL]
        .describe()
    )

    return df


# ============================================================
# Exact 80 / 10 / 10 split
# ============================================================

def split_data(df):

    train_df, temp_df = (
        train_test_split(
            df,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

    val_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.50,
            random_state=RANDOM_STATE,
        )
    )

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .reset_index(drop=True)
    )

    print("\n==============================")
    print("DATA SPLIT")
    print("==============================")

    print(
        "Train:",
        len(train_df),
    )

    print(
        "Validation:",
        len(val_df),
    )

    print(
        "Test:",
        len(test_df),
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# Generate overlapping k-mer text
# ============================================================

def add_kmer_features(df):

    df = df.copy()

    df["kmer_text"] = (
        df["clean_sequence"]
        .apply(
            lambda seq:
            sequence_to_kmer_text(
                seq,
                K,
            )
        )
    )

    before = len(df)

    df = (
        df
        .dropna(
            subset=[
                "kmer_text",
                TARGET_COL,
            ]
        )
        .reset_index(drop=True)
    )

    skipped = (
        before - len(df)
    )

    if skipped > 0:

        print(
            f"Warning: {skipped} rows "
            "removed during k-mer generation."
        )

    return df


# ============================================================
# Regression metrics
# ============================================================

def calculate_regression_metrics(
    y_true,
    y_pred,
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    pearson = pearsonr(
        y_true,
        y_pred,
    ).statistic

    spearman = spearmanr(
        y_true,
        y_pred,
    ).statistic

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


def print_regression_metrics(
    name,
    metrics,
):

    print("\n==============================")
    print(name)
    print("==============================")

    print(
        f"RMSE:     "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"R2:       "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"Pearson:  "
        f"{metrics['pearson']:.4f}"
    )

    print(
        f"Spearman: "
        f"{metrics['spearman']:.4f}"
    )


# ============================================================
# Favorable-range screening metrics
# ============================================================

def compute_screening_metrics(
    y_true,
    y_pred,
    pred_low,
    pred_high,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    true_favorable = (
        (y_true >= TRUE_RANGE_LOW)
        & (y_true <= TRUE_RANGE_HIGH)
    )

    selected = (
        (y_pred >= pred_low)
        & (y_pred <= pred_high)
    )

    tp = int(
        np.sum(
            true_favorable
            & selected
        )
    )

    fp = int(
        np.sum(
            ~true_favorable
            & selected
        )
    )

    fn = int(
        np.sum(
            true_favorable
            & ~selected
        )
    )

    tn = int(
        np.sum(
            ~true_favorable
            & ~selected
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    baseline_prevalence = float(
        true_favorable.mean()
    )

    enrichment = (
        precision
        / baseline_prevalence
        if baseline_prevalence > 0
        else np.nan
    )

    selected_count = int(
        selected.sum()
    )

    selected_fraction = float(
        selected.mean()
    )

    return {
        "pred_low": float(
            pred_low
        ),
        "pred_high": float(
            pred_high
        ),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "baseline_prevalence": (
            baseline_prevalence
        ),

        "enrichment": float(
            enrichment
        ),

        "true_favorable_count": int(
            true_favorable.sum()
        ),

        "selected_count": (
            selected_count
        ),

        "selected_fraction": (
            selected_fraction
        ),
    }


# ============================================================
# Select screening interval on VALIDATION only
# ============================================================

def select_prediction_interval(
    y_true,
    y_pred,
):

    print("\n==============================")
    print("VALIDATION INTERVAL SEARCH")
    print("==============================")

    print(
        "True favorable PNI range:",
        f"{TRUE_RANGE_LOW:.1f}"
        f"-{TRUE_RANGE_HIGH:.1f}",
    )

    print(
        "Minimum required recall:",
        f"{MIN_RECALL:.0%}",
    )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    grid_min = np.floor(
        y_pred.min()
    )

    grid_max = np.ceil(
        y_pred.max()
    )

    grid = np.arange(
        grid_min,
        grid_max + INTERVAL_STEP,
        INTERVAL_STEP,
    )

    results = []

    for i, pred_low in enumerate(
        grid
    ):

        for pred_high in grid[
            i + 1:
        ]:

            metrics = (
                compute_screening_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    pred_low=pred_low,
                    pred_high=pred_high,
                )
            )

            results.append(
                metrics
            )

    scan_df = pd.DataFrame(
        results
    )

    eligible = scan_df[
        scan_df["recall"]
        >= MIN_RECALL
    ].copy()

    if len(eligible) == 0:

        raise RuntimeError(
            "No prediction interval "
            "reached recall >= "
            f"{MIN_RECALL:.2f}"
        )

    eligible = (
        eligible
        .sort_values(
            by=[
                "enrichment",
                "recall",
                "selected_count",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )

    best = (
        eligible
        .iloc[0]
        .to_dict()
    )

    print(
        "\nSelected prediction interval:"
    )

    print(
        f"{best['pred_low']:.2f}"
        f" - "
        f"{best['pred_high']:.2f}"
    )

    print(
        "\nValidation screening metrics:"
    )

    print(
        f"Recall:              "
        f"{best['recall']:.4f}"
    )

    print(
        f"Precision:           "
        f"{best['precision']:.4f}"
    )

    print(
        f"Baseline prevalence: "
        f"{best['baseline_prevalence']:.4f}"
    )

    print(
        f"Enrichment:          "
        f"{best['enrichment']:.2f}x"
    )

    print(
        f"Selected candidates: "
        f"{int(best['selected_count'])}"
    )

    print(
        f"Candidate pool kept: "
        f"{best['selected_fraction']:.2%}"
    )

    return (
        best,
        scan_df,
    )


# ============================================================
# Add screening labels
# ============================================================

def annotate_screening(
    df,
    pred_low,
    pred_high,
):

    result = df.copy()

    result[
        "true_favorable"
    ] = (
        (
            result[TARGET_COL]
            >= TRUE_RANGE_LOW
        )
        & (
            result[TARGET_COL]
            <= TRUE_RANGE_HIGH
        )
    )

    result[
        "selected_for_experiment"
    ] = (
        (
            result[
                "predicted_expression"
            ]
            >= pred_low
        )
        & (
            result[
                "predicted_expression"
            ]
            <= pred_high
        )
    )

    return result


# ============================================================
# Final TEST screening evaluation
# ============================================================

def evaluate_test_screening(
    test_df,
    pred_low,
    pred_high,
):

    print("\n==============================")
    print("FINAL TEST SCREENING RESULTS")
    print("==============================")

    metrics = (
        compute_screening_metrics(
            y_true=(
                test_df[TARGET_COL]
                .to_numpy()
            ),
            y_pred=(
                test_df[
                    "predicted_expression"
                ]
                .to_numpy()
            ),
            pred_low=pred_low,
            pred_high=pred_high,
        )
    )

    print(
        "\nTrue favorable PNI range:"
    )

    print(
        f"{TRUE_RANGE_LOW:.1f}"
        f" - "
        f"{TRUE_RANGE_HIGH:.1f}"
    )

    print(
        "\nPrediction screening range "
        "(selected on validation):"
    )

    print(
        f"{pred_low:.2f}"
        f" - "
        f"{pred_high:.2f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        "\n"
        "                   "
        "Pred Reject   Pred Select"
    )

    print(
        f"True Unfavorable   "
        f"{metrics['tn']:11d}   "
        f"{metrics['fp']:11d}"
    )

    print(
        f"True Favorable     "
        f"{metrics['fn']:11d}   "
        f"{metrics['tp']:11d}"
    )

    print(
        "\nMetrics:"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"F1:                  "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Baseline prevalence: "
        f"{metrics['baseline_prevalence']:.4f}"
    )

    print(
        f"Enrichment:          "
        f"{metrics['enrichment']:.2f}x"
    )

    print(
        "\nTrue favorable sequences:"
    )

    print(
        metrics[
            "true_favorable_count"
        ]
    )

    print(
        "\nSequences selected "
        "for experiment:"
    )

    print(
        metrics[
            "selected_count"
        ]
    )

    print(
        "\nCandidate pool retained:"
    )

    print(
        f"{metrics['selected_fraction']:.2%}"
    )

    print(
        "\nCandidate pool reduced:"
    )

    print(
        f"{1.0 - metrics['selected_fraction']:.2%}"
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load full dataset
    # --------------------------------------------------------

    df = load_cambray_data()

    # --------------------------------------------------------
    # Same split as final CodonBERT
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
    ) = split_data(df)

    # --------------------------------------------------------
    # Generate overlapping 7-mers
    # --------------------------------------------------------

    print("\n==============================")
    print("GENERATING 7-MERS")
    print("==============================")

    train_df = add_kmer_features(
        train_df
    )

    val_df = add_kmer_features(
        val_df
    )

    test_df = add_kmer_features(
        test_df
    )

    print(
        "\nExample sequence:"
    )

    print(
        train_df[
            "clean_sequence"
        ].iloc[0]
    )

    print(
        "\nExample 7-mer text:"
    )

    print(
        train_df[
            "kmer_text"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Inputs / targets
    # --------------------------------------------------------

    X_train_text = (
        train_df[
            "kmer_text"
        ]
        .values
    )

    X_val_text = (
        val_df[
            "kmer_text"
        ]
        .values
    )

    X_test_text = (
        test_df[
            "kmer_text"
        ]
        .values
    )

    y_train = (
        train_df[TARGET_COL]
        .astype(float)
        .values
    )

    y_val = (
        val_df[TARGET_COL]
        .astype(float)
        .values
    )

    y_test = (
        test_df[TARGET_COL]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # TF-IDF
    #
    # Fit vocabulary ONLY on training data.
    # --------------------------------------------------------

    print("\n==============================")
    print("TF-IDF")
    print("==============================")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )

    X_train = (
        vectorizer
        .fit_transform(
            X_train_text
        )
    )

    X_val = (
        vectorizer
        .transform(
            X_val_text
        )
    )

    X_test = (
        vectorizer
        .transform(
            X_test_text
        )
    )

    print(
        "Train feature shape:",
        X_train.shape,
    )

    print(
        "Validation feature shape:",
        X_val.shape,
    )

    print(
        "Test feature shape:",
        X_test.shape,
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer
            .get_feature_names_out()
        ),
    )

    # --------------------------------------------------------
    # Train Ridge
    # --------------------------------------------------------

    print("\n==============================")
    print("TRAINING RIDGE")
    print("==============================")

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Regression predictions
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )

    val_metrics = (
        calculate_regression_metrics(
            y_val,
            val_pred,
        )
    )

    test_metrics = (
        calculate_regression_metrics(
            y_test,
            test_pred,
        )
    )

    print_regression_metrics(
        (
            "VALIDATION: "
            f"{K}-mer TF-IDF + "
            f"Ridge(alpha={RIDGE_ALPHA})"
        ),
        val_metrics,
    )

    print_regression_metrics(
        (
            "TEST: "
            f"{K}-mer TF-IDF + "
            f"Ridge(alpha={RIDGE_ALPHA})"
        ),
        test_metrics,
    )

    # --------------------------------------------------------
    # Attach predictions to dataframes
    # --------------------------------------------------------

    val_df = val_df.copy()

    val_df[
        "predicted_expression"
    ] = val_pred

    test_df = test_df.copy()

    test_df[
        "predicted_expression"
    ] = test_pred

    # --------------------------------------------------------
    # Validation-only screening interval selection
    # --------------------------------------------------------

    (
        best_interval,
        interval_scan,
    ) = select_prediction_interval(
        y_true=y_val,
        y_pred=val_pred,
    )

    pred_low = float(
        best_interval[
            "pred_low"
        ]
    )

    pred_high = float(
        best_interval[
            "pred_high"
        ]
    )

    val_df = annotate_screening(
        val_df,
        pred_low,
        pred_high,
    )

    # --------------------------------------------------------
    # Final test screening evaluation
    # --------------------------------------------------------

    screening_metrics = (
        evaluate_test_screening(
            test_df,
            pred_low,
            pred_high,
        )
    )

    test_df = annotate_screening(
        test_df,
        pred_low,
        pred_high,
    )

    # --------------------------------------------------------
    # Save regression metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "model": (
                    f"{K}-mer TF-IDF + Ridge"
                ),
                "split": "validation",
                "k": K,
                "ridge_alpha": RIDGE_ALPHA,
                **val_metrics,
            },
            {
                "model": (
                    f"{K}-mer TF-IDF + Ridge"
                ),
                "split": "test",
                "k": K,
                "ridge_alpha": RIDGE_ALPHA,
                **test_metrics,
            },
        ]
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    val_df[
        [
            SEQ_COL,
            TARGET_COL,
            "predicted_expression",
            "true_favorable",
            "selected_for_experiment",
        ]
    ].to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    test_df[
        [
            SEQ_COL,
            TARGET_COL,
            "predicted_expression",
            "true_favorable",
            "selected_for_experiment",
        ]
    ].to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save interval search
    # --------------------------------------------------------

    interval_scan.to_csv(
        OUTPUT_DIR
        / "validation_favorable_interval_scan.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save screening summary
    # --------------------------------------------------------

    summary = {

        "model": (
            f"{K}-mer TF-IDF + Ridge"
        ),

        "ridge_alpha": (
            RIDGE_ALPHA
        ),

        "true_favorable_range": {
            "low": TRUE_RANGE_LOW,
            "high": TRUE_RANGE_HIGH,
        },

        "minimum_validation_recall": (
            MIN_RECALL
        ),

        "selected_prediction_interval": {
            "low": pred_low,
            "high": pred_high,
        },

        "validation_recall": float(
            best_interval[
                "recall"
            ]
        ),

        "validation_precision": float(
            best_interval[
                "precision"
            ]
        ),

        "validation_enrichment": float(
            best_interval[
                "enrichment"
            ]
        ),

        "test_recall": (
            screening_metrics[
                "recall"
            ]
        ),

        "test_precision": (
            screening_metrics[
                "precision"
            ]
        ),

        "test_f1": (
            screening_metrics[
                "f1"
            ]
        ),

        "test_baseline_prevalence": (
            screening_metrics[
                "baseline_prevalence"
            ]
        ),

        "test_enrichment": (
            screening_metrics[
                "enrichment"
            ]
        ),

        "test_selected_count": (
            screening_metrics[
                "selected_count"
            ]
        ),

        "test_selected_fraction": (
            screening_metrics[
                "selected_fraction"
            ]
        ),

        "test_true_favorable_count": (
            screening_metrics[
                "true_favorable_count"
            ]
        ),
    }

    with open(
        OUTPUT_DIR
        / "favorable_screening_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n==============================")
    print("SAVED RESULTS")
    print("==============================")

    print(
        OUTPUT_DIR
    )

    print(
        "\nSaved files:"
    )

    print(
        "- metrics.csv"
    )

    print(
        "- validation_predictions.csv"
    )

    print(
        "- test_predictions.csv"
    )

    print(
        "- validation_favorable_interval_scan.csv"
    )

    print(
        "- favorable_screening_summary.json"
    )


if __name__ == "__main__":
    main()