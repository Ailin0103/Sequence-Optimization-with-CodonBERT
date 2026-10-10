from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
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
# Favorable expression screening definition
# ============================================================

# From the Cambray dataset:
# cellular fitness is near its population-level maximum
# over roughly this protein-expression range.
TRUE_RANGE_LOW = 40.0
TRUE_RANGE_HIGH = 60.0

# When selecting the prediction interval on VALIDATION,
# require at least this much recall among truly favorable
# sequences.
MIN_RECALL = 0.80

# Search prediction thresholds at 1 PNI-unit resolution.
INTERVAL_STEP = 1.0


# ============================================================
# Sequence preprocessing
# ============================================================

def format_sequence(seq):

    if pd.isna(seq):
        return None

    seq = str(seq).upper().strip()

    # DNA -> RNA
    seq = seq.replace("T", "U")

    if any(
        base not in {"A", "U", "G", "C"}
        for base in seq
    ):
        return None

    if len(seq) % 3 != 0:
        return None

    codons = [
        seq[i:i + 3]
        for i in range(0, len(seq), 3)
    ]

    # Leave room for [CLS] and [SEP]
    if len(codons) > MAX_LENGTH - 2:
        return None

    return " ".join(codons)


# ============================================================
# Load data and reconstruct exact train/val/test split
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
    # EXACT SAME split used during final model training
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

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Inspect training target distribution
    # --------------------------------------------------------

    print("\n==============================")
    print("TRAINING TARGET DISTRIBUTION")
    print("==============================")

    print(
        train_df[TARGET_COL].describe(
            percentiles=[
                0.10,
                0.20,
                0.25,
                0.50,
                0.75,
                0.80,
                0.90,
            ]
        )
    )

    train_true_favorable = (
        (train_df[TARGET_COL] >= TRUE_RANGE_LOW)
        & (train_df[TARGET_COL] <= TRUE_RANGE_HIGH)
    )

    print(
        "\nTraining fraction in "
        f"{TRUE_RANGE_LOW:.0f}-{TRUE_RANGE_HIGH:.0f} "
        "favorable range:"
    )

    print(
        f"{train_true_favorable.mean():.4%}"
    )

    print(
        "Training favorable sequences:",
        int(train_true_favorable.sum()),
    )

    print("\nSplit:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return (
        val_df,
        test_df,
    )


# ============================================================
# Load saved final CodonBERT model
# ============================================================

def load_model():

    print("\n==============================")
    print("LOADING FINAL MODEL")
    print("==============================")

    print(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = (
        BertForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    model.to(device)

    print("Device:", device)

    # --------------------------------------------------------
    # Load target standardization parameters
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
# Batched protein-expression prediction
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
            for key, value
            in inputs.items()
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
        # back to original PNI scale.
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
# Favorable-range screening metrics
# ============================================================

def compute_screening_metrics(
    y_true,
    y_pred,
    pred_low,
    pred_high,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --------------------------------------------------------
    # Ground-truth desirable sequences:
    #
    # actual PNI lies inside the fitness-favorable
    # protein-expression range.
    # --------------------------------------------------------

    true_favorable = (
        (y_true >= TRUE_RANGE_LOW)
        & (y_true <= TRUE_RANGE_HIGH)
    )

    # --------------------------------------------------------
    # Model-selected sequences
    # --------------------------------------------------------

    predicted_favorable = (
        (y_pred >= pred_low)
        & (y_pred <= pred_high)
    )

    tp = int(
        np.sum(
            true_favorable
            & predicted_favorable
        )
    )

    fp = int(
        np.sum(
            ~true_favorable
            & predicted_favorable
        )
    )

    fn = int(
        np.sum(
            true_favorable
            & ~predicted_favorable
        )
    )

    tn = int(
        np.sum(
            ~true_favorable
            & ~predicted_favorable
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
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    baseline_prevalence = float(
        true_favorable.mean()
    )

    enrichment = (
        precision / baseline_prevalence
        if baseline_prevalence > 0
        else np.nan
    )

    selected_count = int(
        predicted_favorable.sum()
    )

    selected_fraction = float(
        predicted_favorable.mean()
    )

    return {

        "pred_low": float(pred_low),
        "pred_high": float(pred_high),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

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
# Select prediction interval using VALIDATION only
# ============================================================

def select_prediction_interval(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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

    # --------------------------------------------------------
    # Search possible prediction intervals
    # --------------------------------------------------------

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

    for i, pred_low in enumerate(grid):

        for pred_high in grid[i + 1:]:

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

    # --------------------------------------------------------
    # Only consider intervals with sufficient recall
    # --------------------------------------------------------

    eligible = scan_df[
        scan_df["recall"] >= MIN_RECALL
    ].copy()

    if len(eligible) == 0:

        raise RuntimeError(
            "No prediction interval reached "
            f"recall >= {MIN_RECALL:.2f}"
        )

    # --------------------------------------------------------
    # Primary objective:
    # maximize enrichment.
    #
    # Since baseline prevalence is fixed,
    # maximizing enrichment is equivalent to
    # maximizing precision.
    #
    # Tie-break:
    # 1. higher recall
    # 2. fewer selected experiments
    # --------------------------------------------------------

    eligible = eligible.sort_values(
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
        "\nValidation metrics:"
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
        f"F1:                  "
        f"{best['f1']:.4f}"
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
# Add screening labels to dataframe
# ============================================================

def annotate_screening_flags(
    df,
    pred_low,
    pred_high,
):

    result = df.copy()

    result[
        "true_favorable"
    ] = (
        (result[TARGET_COL] >= TRUE_RANGE_LOW)
        & (
            result[TARGET_COL]
            <= TRUE_RANGE_HIGH
        )
    )

    result[
        "selected_for_experiment"
    ] = (
        (
            result["predicted_expression"]
            >= pred_low
        )
        & (
            result["predicted_expression"]
            <= pred_high
        )
    )

    return result


# ============================================================
# Evaluate frozen screening rule on TEST
# ============================================================

def evaluate_test(
    test_df,
    pred_low,
    pred_high,
):

    print("\n==============================")
    print("FINAL TEST SCREENING RESULTS")
    print("==============================")

    y_true = (
        test_df[TARGET_COL]
        .to_numpy()
    )

    y_pred = (
        test_df[
            "predicted_expression"
        ]
        .to_numpy()
    )

    metrics = (
        compute_screening_metrics(
            y_true=y_true,
            y_pred=y_pred,
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
    # Data
    # --------------------------------------------------------

    val_df, test_df = (
        load_data()
    )

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
    # Select screening interval ONLY from validation
    # --------------------------------------------------------

    best_interval, interval_scan = (
        select_prediction_interval(
            y_true=(
                val_df[TARGET_COL]
                .to_numpy()
            ),
            y_pred=(
                val_df[
                    "predicted_expression"
                ]
                .to_numpy()
            ),
        )
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

    val_df = annotate_screening_flags(
        df=val_df,
        pred_low=pred_low,
        pred_high=pred_high,
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
    # Final TEST evaluation using frozen validation interval
    # --------------------------------------------------------

    test_metrics = evaluate_test(
        test_df=test_df,
        pred_low=pred_low,
        pred_high=pred_high,
    )

    test_df = annotate_screening_flags(
        df=test_df,
        pred_low=pred_low,
        pred_high=pred_high,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    interval_scan.to_csv(
        OUTPUT_DIR
        / "validation_favorable_interval_scan.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR
        / "validation_favorable_predictions.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR
        / "test_favorable_screening_predictions.csv",
        index=False,
    )

    summary = {

        "true_favorable_range": {
            "low": TRUE_RANGE_LOW,
            "high": TRUE_RANGE_HIGH,
        },

        "minimum_validation_recall": (
            MIN_RECALL
        ),

        "interval_search_step": (
            INTERVAL_STEP
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

        "validation_selected_count": int(
            best_interval[
                "selected_count"
            ]
        ),

        "test_recall": (
            test_metrics[
                "recall"
            ]
        ),

        "test_precision": (
            test_metrics[
                "precision"
            ]
        ),

        "test_f1": (
            test_metrics[
                "f1"
            ]
        ),

        "test_baseline_prevalence": (
            test_metrics[
                "baseline_prevalence"
            ]
        ),

        "test_enrichment": (
            test_metrics[
                "enrichment"
            ]
        ),

        "test_tp": (
            test_metrics[
                "tp"
            ]
        ),

        "test_fn": (
            test_metrics[
                "fn"
            ]
        ),

        "test_fp": (
            test_metrics[
                "fp"
            ]
        ),

        "test_tn": (
            test_metrics[
                "tn"
            ]
        ),

        "test_selected_count": (
            test_metrics[
                "selected_count"
            ]
        ),

        "test_selected_fraction": (
            test_metrics[
                "selected_fraction"
            ]
        ),

        "test_true_favorable_count": (
            test_metrics[
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
    print("SAVED")
    print("==============================")

    print(OUTPUT_DIR)

    print(
        "\nSaved files:"
    )

    print(
        "- validation_favorable_interval_scan.csv"
    )

    print(
        "- validation_favorable_predictions.csv"
    )

    print(
        "- test_favorable_screening_predictions.csv"
    )

    print(
        "- favorable_screening_summary.json"
    )


if __name__ == "__main__":
    main()