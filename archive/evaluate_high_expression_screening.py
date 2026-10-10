"""
Evaluate the fine-tuned CodonBERT model as a high-expression
screening tool.

Goal
----
Use the regression model to identify high-expression sequences
while prioritizing recall.

Workflow
--------
1. Reconstruct the same validation/test split used during training.
2. Run the saved fine-tuned CodonBERT model on validation data.
3. Define "true high expression" using an observed-expression cutoff.
4. Search validation prediction thresholds.
5. Select the highest threshold that achieves the target recall.
   This gives the best precision while satisfying the recall requirement.
6. Freeze that threshold.
7. Evaluate once on the held-out test set.

No model retraining is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)


# ============================================================
# Config
# ============================================================

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
)

MODEL_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "cambray_codonbert_regression"
    / "final_model"
)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "cambray_codonbert_regression"
    / "screening_analysis"
)

SEQ_COL = "gs.sequence"
TARGET_COL = "clean.lin.prot.mean"

RANDOM_STATE = 42

MAX_LENGTH = 64
BATCH_SIZE = 64


# ============================================================
# Screening definition
# ============================================================

# Biological definition of a truly high-expression sequence.
#
# Example:
# true protein expression >= 80
# is considered "high expression".
#
# We can change this later if we prefer top 10%, top 5%, etc.

TRUE_HIGH_THRESHOLD = 80.0


# Screening objective:
# recover at least 95% of truly high-expression sequences.

TARGET_RECALL = 0.95


# ============================================================
# Sequence preprocessing
# ============================================================

def format_sequence(seq):

    if pd.isna(seq):
        return None

    seq = str(seq).upper().strip()

    # DNA -> RNA
    seq = seq.replace("T", "U")

    if any(base not in {"A", "U", "G", "C"} for base in seq):
        return None

    if len(seq) % 3 != 0:
        return None

    codons = [
        seq[i:i + 3]
        for i in range(0, len(seq), 3)
    ]

    if len(codons) > MAX_LENGTH - 2:
        return None

    return " ".join(codons)


# ============================================================
# Load data and reconstruct split
# ============================================================

def load_data():

    print("\n==============================")
    print("LOADING DATA")
    print("==============================")

    df = pd.read_csv(DATA_PATH)

    print("Original rows:", len(df))

    df = df[
        [
            SEQ_COL,
            TARGET_COL,
        ]
    ].copy()

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

    df["formatted_seq"] = (
        df[SEQ_COL]
        .apply(format_sequence)
    )

    df = (
        df
        .dropna(subset=["formatted_seq"])
        .reset_index(drop=True)
    )

    print("Valid rows:", len(df))

    # --------------------------------------------------------
    # EXACT SAME split procedure used during final training
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )
    print("\n==============================")
    print("TRAINING TARGET DISTRIBUTION")
    print("==============================")

    print(
        train_df[TARGET_COL].describe(
            percentiles=[
                0.80,
                0.90,
                0.95,
                0.975,
                0.99,
            ]
        )
    )

    fraction_high = (
        train_df[TARGET_COL] >= 80
    ).mean()

    print(
        "\nFraction with expression >= 80:"
    )

    print(
        f"{fraction_high:.4%}"
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nSplit:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return (
        val_df,
        test_df,
    )


# ============================================================
# Load saved model
# ============================================================

def load_model():

    print("\n==============================")
    print("LOADING FINAL MODEL")
    print("==============================")

    print(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    model.to(device)

    print("Device:", device)

    # --------------------------------------------------------
    # Load target scaler
    # --------------------------------------------------------

    scaler_path = (
        MODEL_DIR
        / "target_scaler.json"
    )

    with open(
        scaler_path,
        "r",
    ) as f:
        scaler = json.load(f)

    target_mean = float(
        scaler["target_mean"]
    )

    target_std = float(
        scaler["target_std"]
    )

    print(
        "Target mean:",
        target_mean,
    )

    print(
        "Target std:",
        target_std,
    )

    return (
        tokenizer,
        model,
        device,
        target_mean,
        target_std,
    )


# ============================================================
# Batched prediction
# ============================================================

def predict_dataframe(
    df,
    tokenizer,
    model,
    device,
    target_mean,
    target_std,
):

    sequences = (
        df["formatted_seq"]
        .tolist()
    )

    predictions = []

    print(
        f"\nPredicting {len(sequences)} sequences..."
    )

    for start in range(
        0,
        len(sequences),
        BATCH_SIZE,
    ):

        batch_sequences = sequences[
            start:start + BATCH_SIZE
        ]

        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        pred_z = (
            outputs
            .logits
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

        # Convert standardized predictions
        # back to original expression scale
        pred_y = (
            pred_z * target_std
            + target_mean
        )

        predictions.extend(
            pred_y.tolist()
        )

    result = df.copy()

    result[
        "predicted_expression"
    ] = predictions

    return result


# ============================================================
# Binary screening metrics
# ============================================================

def calculate_screening_metrics(
    true_expression,
    predicted_expression,
    prediction_threshold,
):

    true_high = (
        true_expression
        >= TRUE_HIGH_THRESHOLD
    )

    pred_high = (
        predicted_expression
        >= prediction_threshold
    )

    cm = confusion_matrix(
        true_high,
        pred_high,
        labels=[
            False,
            True,
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        true_high,
        pred_high,
        zero_division=0,
    )

    recall = recall_score(
        true_high,
        pred_high,
        zero_division=0,
    )

    f1 = f1_score(
        true_high,
        pred_high,
        zero_division=0,
    )

    selected = int(
        pred_high.sum()
    )

    true_positive_count = int(
        true_high.sum()
    )

    return {
        "threshold": float(
            prediction_threshold
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "selected": selected,
        "true_high_count": true_positive_count,
    }


# ============================================================
# Select threshold using VALIDATION only
# ============================================================

def choose_validation_threshold(
    val_df,
):

    print("\n==============================")
    print("VALIDATION THRESHOLD SEARCH")
    print("==============================")

    true_expression = (
        val_df[TARGET_COL]
        .values
    )

    predicted_expression = (
        val_df["predicted_expression"]
        .values
    )

    # --------------------------------------------------------
    # Candidate thresholds:
    # use every unique predicted value.
    # --------------------------------------------------------

    candidate_thresholds = np.sort(
        np.unique(
            predicted_expression
        )
    )

    results = []

    for threshold in candidate_thresholds:

        metrics = calculate_screening_metrics(
            true_expression=true_expression,
            predicted_expression=predicted_expression,
            prediction_threshold=threshold,
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Only thresholds satisfying recall requirement
    # --------------------------------------------------------

    eligible = results_df[
        results_df["recall"]
        >= TARGET_RECALL
    ].copy()

    if len(eligible) == 0:

        raise ValueError(
            "No validation threshold achieved "
            f"target recall >= {TARGET_RECALL:.3f}"
        )

    # --------------------------------------------------------
    # Among thresholds meeting target recall,
    # choose the one with highest precision.
    #
    # If tied, prefer the higher threshold.
    # --------------------------------------------------------

    eligible = eligible.sort_values(
        by=[
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = eligible.iloc[0]

    print(
        f"\nTrue high-expression definition:"
    )

    print(
        f"Observed expression >= "
        f"{TRUE_HIGH_THRESHOLD:.2f}"
    )

    print(
        f"\nTarget validation recall:"
    )

    print(
        f"{TARGET_RECALL:.1%}"
    )

    print(
        "\nSelected prediction threshold:"
    )

    print(
        f"{best['threshold']:.4f}"
    )

    print(
        "\nValidation performance:"
    )

    print(
        f"Recall:    "
        f"{best['recall']:.4f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"F1:        "
        f"{best['f1']:.4f}"
    )

    print(
        f"TP: {int(best['tp'])}"
    )

    print(
        f"FN: {int(best['fn'])}"
    )

    print(
        f"FP: {int(best['fp'])}"
    )

    print(
        f"TN: {int(best['tn'])}"
    )

    print(
        f"Sequences selected: "
        f"{int(best['selected'])}"
    )

    return (
        float(best["threshold"]),
        results_df,
    )


# ============================================================
# Evaluate frozen threshold on TEST
# ============================================================

def evaluate_test(
    test_df,
    prediction_threshold,
):

    print("\n==============================")
    print("FINAL TEST SCREENING RESULTS")
    print("==============================")

    metrics = calculate_screening_metrics(
        true_expression=(
            test_df[TARGET_COL]
            .values
        ),
        predicted_expression=(
            test_df[
                "predicted_expression"
            ].values
        ),
        prediction_threshold=(
            prediction_threshold
        ),
    )

    print(
        "\nPrediction threshold "
        "(selected on validation):"
    )

    print(
        f"{prediction_threshold:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        "\n             Pred Low   Pred High"
    )

    print(
        f"True Low     "
        f"{metrics['tn']:8d}   "
        f"{metrics['fp']:9d}"
    )

    print(
        f"True High    "
        f"{metrics['fn']:8d}   "
        f"{metrics['tp']:9d}"
    )

    print(
        "\nMetrics:"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"\nTrue high-expression sequences: "
        f"{metrics['true_high_count']}"
    )

    print(
        f"Sequences selected for experiment: "
        f"{metrics['selected']}"
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
    # Data
    # --------------------------------------------------------

    val_df, test_df = load_data()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    (
        tokenizer,
        model,
        device,
        target_mean,
        target_std,
    ) = load_model()

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print("\n==============================")
    print("VALIDATION PREDICTIONS")
    print("==============================")

    val_df = predict_dataframe(
        df=val_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    # --------------------------------------------------------
    # Select threshold ONLY from validation
    # --------------------------------------------------------

    (
        selected_threshold,
        threshold_results,
    ) = choose_validation_threshold(
        val_df
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    print("\n==============================")
    print("TEST PREDICTIONS")
    print("==============================")

    test_df = predict_dataframe(
        df=test_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_metrics = evaluate_test(
        test_df=test_df,
        prediction_threshold=selected_threshold,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    threshold_results.to_csv(
        OUTPUT_DIR
        / "validation_threshold_scan.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR
        / "test_predictions_screening.csv",
        index=False,
    )

    summary = {
        "true_high_threshold": (
            TRUE_HIGH_THRESHOLD
        ),
        "target_validation_recall": (
            TARGET_RECALL
        ),
        "selected_prediction_threshold": (
            selected_threshold
        ),
        "test_recall": (
            test_metrics["recall"]
        ),
        "test_precision": (
            test_metrics["precision"]
        ),
        "test_f1": (
            test_metrics["f1"]
        ),
        "test_tp": (
            test_metrics["tp"]
        ),
        "test_fn": (
            test_metrics["fn"]
        ),
        "test_fp": (
            test_metrics["fp"]
        ),
        "test_tn": (
            test_metrics["tn"]
        ),
        "test_selected": (
            test_metrics["selected"]
        ),
        "test_true_high_count": (
            test_metrics[
                "true_high_count"
            ]
        ),
    }

    with open(
        OUTPUT_DIR
        / "screening_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n==============================")
    print("SAVED")
    print("==============================")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()