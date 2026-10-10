from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)

import json
# ============================================================
# Config
# ============================================================

MODEL_NAME = "lhallee/CodonBERT"

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "cambray_codonbert_regression"
)

SEQ_COL = "gs.sequence"
TARGET_COL = "clean.lin.prot.mean"

RANDOM_STATE = 42

# Cambray designed sequence = 96 nt = 32 codons.
# + [CLS] and [SEP], so 64 is more than enough.
MAX_LENGTH = 64

# First experiment:
# use 10k to compare with previous baseline experiments.
# Set to None later for the full dataset.
SAMPLE_N = None

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# Sequence preprocessing
# ============================================================

def codon_tokenize(seq):
    """
    Convert DNA coding sequence:

        ATGGCCUUU...

    to RNA codon tokens:

        AUG GCC UUU ...

    One non-overlapping codon = one CodonBERT token.
    """

    if pd.isna(seq):
        return None

    seq = str(seq).upper().strip()

    # DNA -> RNA representation
    seq = seq.replace("T", "U")

    # Only standard RNA nucleotides
    if any(base not in {"A", "U", "G", "C"} for base in seq):
        return None

    # Must be frame-aligned
    if len(seq) % 3 != 0:
        return None

    codons = [
        seq[i:i + 3]
        for i in range(0, len(seq), 3)
    ]

    # We do NOT want silent truncation for this dataset.
    if len(codons) > MAX_LENGTH - 2:
        raise ValueError(
            f"Sequence contains {len(codons)} codons, "
            f"but MAX_LENGTH={MAX_LENGTH}. "
            "Inspect sequence lengths before training."
        )

    return " ".join(codons)


# ============================================================
# Load and clean Cambray data
# ============================================================

def load_cambray_data():

    print("\nLoading:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print("\nOriginal rows:", len(df))

    # Keep only columns needed here
    df = df[
        [
            SEQ_COL,
            TARGET_COL,
        ]
    ].copy()

    # Remove missing target / sequence
    df = df.dropna(
        subset=[
            SEQ_COL,
            TARGET_COL,
        ]
    )

    # Ensure regression target is numeric
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce",
    )

    df = df.dropna(
        subset=[TARGET_COL]
    )

    # --------------------------------------------------------
    # Sequence length sanity check
    # --------------------------------------------------------

    lengths = (
        df[SEQ_COL]
        .astype(str)
        .str.len()
    )

    print("\nSequence length summary (nt):")
    print(lengths.describe())

    print("\nMost common sequence lengths:")
    print(
        lengths
        .value_counts()
        .head(10)
    )

    # --------------------------------------------------------
    # Codon formatting
    # --------------------------------------------------------

    df["formatted_seq"] = (
        df[SEQ_COL]
        .apply(codon_tokenize)
    )

    before = len(df)

    df = df.dropna(
        subset=["formatted_seq"]
    ).reset_index(drop=True)

    print(
        "\nValid sequences:",
        len(df),
    )

    print(
        "Skipped sequences:",
        before - len(df),
    )

    # --------------------------------------------------------
    # Optional 10k sample for first experiment
    # --------------------------------------------------------

    if SAMPLE_N is not None:

        if SAMPLE_N > len(df):
            raise ValueError(
                f"SAMPLE_N={SAMPLE_N} "
                f"but only {len(df)} valid rows exist."
            )

        df = df.sample(
            n=SAMPLE_N,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

        print(
            f"\nUsing {len(df)} sampled rows "
            "for this experiment."
        )

    print("\nTarget summary:")
    print(
        df[TARGET_COL].describe()
    )

    return df


# ============================================================
# Train / validation / test split
# ============================================================

def split_data(df):
    """
    Fixed reproducible split:

        80% train
        10% validation
        10% test
    """

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
    print("DATA SPLIT")
    print("==============================")

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# Convert pandas -> Hugging Face Dataset
# ============================================================

def dataframe_to_dataset(df):

    return Dataset.from_dict(
        {
            "seq": df["formatted_seq"].tolist(),

            # IMPORTANT:
            # regression labels must be float
            "label": (
                df["target_z"]
                .astype(float)
                .tolist()
            ),
        }
    )


# ============================================================
# Regression metrics
# ============================================================

def make_compute_metrics(target_mean, target_std):

    def compute_metrics(eval_pred):

        predictions_z, labels_z = eval_pred

        predictions_z = np.asarray(
            predictions_z
        ).reshape(-1)

        labels_z = np.asarray(
            labels_z
        ).reshape(-1)

        # Convert back to original protein-expression scale
        predictions = (
            predictions_z * target_std
            + target_mean
        )

        labels = (
            labels_z * target_std
            + target_mean
        )

        rmse = np.sqrt(
            mean_squared_error(
                labels,
                predictions,
            )
        )

        r2 = r2_score(
            labels,
            predictions,
        )

        pearson = pearsonr(
            labels,
            predictions,
        ).statistic

        spearman = spearmanr(
            labels,
            predictions,
        ).statistic

        return {
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson": float(pearson),
            "spearman": float(spearman),
        }

    return compute_metrics


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    df = load_cambray_data()

    train_df, val_df, test_df = split_data(df)
    # ============================================================
    # Standardize target using TRAINING SET only
    # ============================================================

    target_mean = train_df[TARGET_COL].mean()
    target_std = train_df[TARGET_COL].std()

    print("\nTarget standardization:")
    print("Training mean:", target_mean)
    print("Training std:", target_std)

    train_df["target_z"] = (
        train_df[TARGET_COL] - target_mean
    ) / target_std

    val_df["target_z"] = (
        val_df[TARGET_COL] - target_mean
    ) / target_std

    test_df["target_z"] = (
        test_df[TARGET_COL] - target_mean
    ) / target_std

    train_ds = dataframe_to_dataset(
        train_df
    )

    val_ds = dataframe_to_dataset(
        val_df
    )

    test_ds = dataframe_to_dataset(
        test_df
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\n==============================")
    print("TOKENIZER")
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    example = "AUG GCC UUU"

    encoded = tokenizer(
        example,
        add_special_tokens=True,
    )

    print("Example:", example)

    print(
        "Tokens:",
        tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )
    )

    print(
        "Input IDs:",
        encoded["input_ids"],
    )

    print(
        "Tokenizer size:",
        len(tokenizer),
    )

    # --------------------------------------------------------
    # Tokenize datasets
    # --------------------------------------------------------

    def encode(batch):

        return tokenizer(
            batch["seq"],

            # sequences should already fit;
            # truncation here is only a final safeguard
            truncation=True,

            max_length=MAX_LENGTH,

            # Important:
            # dynamic padding instead of padding all to 64
            padding=False,
        )

    print("\nTokenizing train set...")

    train_ds = train_ds.map(
        encode,
        batched=True,
    )

    print("Tokenizing validation set...")

    val_ds = val_ds.map(
        encode,
        batched=True,
    )

    print("Tokenizing test set...")

    test_ds = test_ds.map(
        encode,
        batched=True,
    )

    # --------------------------------------------------------
    # Token-ID sanity check
    # --------------------------------------------------------

    max_token_id = max(
        max(ids)
        for ids in train_ds["input_ids"]
    )

    print(
        "\nMaximum token ID:",
        max_token_id,
    )

    print(
        "Tokenizer size:",
        len(tokenizer),
    )

    assert max_token_id < len(tokenizer)

    # Dynamic padding:
    # each batch padded only to longest sequence in that batch
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # ============================================================
    # Model
    # ============================================================

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
    )

    # ============================================================
    # Freeze layers
    # ============================================================

    #NUM_TRAINABLE_LAYERS = 2

    #for param in model.bert.parameters():
    #    param.requires_grad = False

    #for layer in model.bert.encoder.layer[-NUM_TRAINABLE_LAYERS:]:
    #    for param in layer.parameters():
    #        param.requires_grad = True

    #if model.bert.pooler is not None:
    #    for param in model.bert.pooler.parameters():
    #        param.requires_grad = True

    #for param in model.classifier.parameters():
    #    param.requires_grad = True


    # ============================================================
    # Check trainable parameters
    # ============================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,} / {total_params:,}"
    )



    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=EPOCHS,

        # Evaluate after every epoch
        eval_strategy="epoch",

        # For the first experiment:
        # don't save/reload "best model",
        # because we saw checkpoint compatibility warnings
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,

        logging_strategy="steps",
        logging_steps=100,

        report_to="none",

        seed=RANDOM_STATE,

        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_ds,
        eval_dataset=val_ds,

        compute_metrics=make_compute_metrics(target_mean, target_std),

        data_collator=data_collator,
    )
    
    
    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n==============================")
    print("CAMBRAY FINE-TUNING")
    print("==============================")

    trainer.train()
    # ============================================================
    # Save final model
    # ============================================================
    FINAL_MODEL_DIR = (
    OUTPUT_DIR
    / "final_model"
    )

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        FINAL_MODEL_DIR
    )

    tokenizer.save_pretrained(
        FINAL_MODEL_DIR
    )

    print(
        "\nSaved final model to:",
        FINAL_MODEL_DIR,
    )
    # ============================================================
    # Save target standardization parameters
    # ============================================================

    target_scaler = {
        "target_column": TARGET_COL,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }

    with open(
        FINAL_MODEL_DIR / "target_scaler.json",
        "w",
    ) as f:
        json.dump(
            target_scaler,
            f,
            indent=2,
        )


    print(
        "\nSaved final model to:",
        FINAL_MODEL_DIR,
    )
    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n==============================")
    print("VALIDATION")
    print("==============================")

    val_metrics = trainer.evaluate(
        val_ds
    )

    print(val_metrics)

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print("\n==============================")
    print("TEST SET")
    print("==============================")

    test_output = trainer.predict(
        test_ds
    )

    print("\nTest metrics:")

    for key, value in (
        test_output.metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Save predictions for later analysis
    # --------------------------------------------------------

    predictions = np.asarray(
        test_output.predictions
    ).reshape(-1)

    labels = np.asarray(
        test_output.label_ids
    ).reshape(-1)

    results_df = pd.DataFrame(
        {
            "true_protein_expression": labels,
            "predicted_protein_expression": predictions,
        }
    )

    results_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        "\nSaved test predictions to:"
    )

    print(results_path)


if __name__ == "__main__":
    main()