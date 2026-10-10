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

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Config
# ============================================================

MODEL_NAME = "lhallee/CodonBERT"

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "codonbert_ecoli_proteins.csv"
)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "ecoli_tiny_overfit"
)

RANDOM_STATE = 42

NUM_LABELS = 3
MAX_LENGTH = 1024

# Tiny overfit experiment
SAMPLES_PER_CLASS = 10

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 5e-5


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
    Convert:
        AUGGCCUUU
    to:
        AUG GCC UUU
    """

    seq = str(seq).upper().replace("T", "U").strip()

    # Must represent complete codons
    if len(seq) % 3 != 0:
        return None

    codons = [
        seq[i:i + 3]
        for i in range(0, len(seq), 3)
    ]

    # Leave space for [CLS] and [SEP]
    if len(codons) > MAX_LENGTH - 2:
        return None

    return " ".join(codons)


# ============================================================
# Build tiny balanced dataset
# ============================================================

def build_tiny_dataset(df):

    # Only official E. coli training samples
    train_df = df[
        (df["Dataset"] == "E.Coli proteins")
        & (df["Split"] == "train")
    ].copy()

    # Take equal number from class 0 / 1 / 2
    tiny_df = (
        train_df
        .groupby("Value", group_keys=False)
        .sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print("\nTiny dataset label counts:")
    print(
        tiny_df["Value"]
        .value_counts()
        .sort_index()
    )

    sequences = []
    labels = []

    skipped = 0

    for seq, label in zip(
        tiny_df["Sequence"],
        tiny_df["Value"],
    ):

        formatted = codon_tokenize(seq)

        if formatted is None:
            skipped += 1
            continue

        sequences.append(formatted)
        labels.append(int(float(label)))

    print(
        f"\nFinal tiny dataset: "
        f"{len(sequences)} samples, "
        f"{skipped} skipped"
    )

    return Dataset.from_dict(
        {
            "seq": sequences,
            "label": labels,
        }
    )



# ============================================================
# Metric
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy
    }


# ============================================================
# Main
# ============================================================

def main():

    print("Loading data:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Tiny balanced training set
    # --------------------------------------------------------

    train_ds = build_tiny_dataset(df)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
    )

    def encode(batch):

        # Important:
        # DO NOT pad every sequence to 1024.
        # Dynamic padding is handled later.
        return tokenizer(
            batch["seq"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    print("\nTokenizing...")

    train_ds = train_ds.map(
        encode,
        batched=True,
    )

    # Pad only to the longest sequence
    # inside the current batch.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading pretrained CodonBERT...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
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

        # This experiment is NOT model selection.
        # We only want to see whether it can memorize.
        save_strategy="no",

        logging_strategy="steps",
        logging_steps=50,

        report_to="none",

        seed=RANDOM_STATE,

        # Avoid irrelevant MPS warning
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_ds,

        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n==============================")
    print("TINY OVERFIT TRAINING")
    print("==============================")

    trainer.train()

    # --------------------------------------------------------
    # Predict on THE SAME TRAINING SET
    # intentionally
    # --------------------------------------------------------

    print("\n==============================")
    print("OVERFIT CHECK ON TRAINING SET")
    print("==============================")

    output = trainer.predict(train_ds)

    predictions = np.argmax(
        output.predictions,
        axis=-1,
    )

    labels = output.label_ids

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    print(
        f"\nTraining accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print("\nPrediction counts:")

    unique, counts = np.unique(
        predictions,
        return_counts=True,
    )

    for cls, count in zip(
        unique,
        counts,
    ):
        print(
            f"Class {cls}: {count}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        labels,
        predictions,
    )

    print(cm)

    # --------------------------------------------------------
    # Full report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()