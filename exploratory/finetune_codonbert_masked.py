"""
Fine-tune CodonBERT on DNA sequences with masked language modeling.
This script freezes all layers except the last 3 transformer layers.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path


class DNASequenceDataset(Dataset):
    """Dataset for DNA sequences with masked language modeling."""

    def __init__(self, sequences, tokenizer, max_length=512, mlm_probability=0.15):
        """
        Parameters
        ----------
        sequences : list of str
            List of DNA sequences
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer for the model
        max_length : int
            Maximum sequence length
        mlm_probability : float
            Probability of masking each token
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels for MLM (clone input_ids)
        labels = input_ids.clone()

        # Create mask for MLM
        # Don't mask special tokens (CLS, SEP, PAD)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # Randomly mask tokens
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time: replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time: replace with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time: keep the original token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CodonBERTWithMLMHead(nn.Module):
    """CodonBERT model with MLM head for fine-tuning."""

    def __init__(
        self,
        model_name="lhallee/CodonBERT",
        freeze_layers=True,
        num_layers_to_unfreeze=3,
    ):
        """
        Parameters
        ----------
        model_name : str
            Name of the pretrained model
        freeze_layers : bool
            Whether to freeze layers
        num_layers_to_unfreeze : int
            Number of final layers to keep unfrozen
        """
        super().__init__()

        # Load pretrained model
        self.bert = AutoModel.from_pretrained(model_name)

        # Get hidden size from config
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size

        # Add MLM head with proper initialization
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        # Initialize weights
        self.mlm_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.mlm_head.bias is not None:
            self.mlm_head.bias.data.zero_()  # Freeze layers except the last few
        if freeze_layers:
            self._freeze_layers(num_layers_to_unfreeze)

    def _freeze_layers(self, num_layers_to_unfreeze):
        """Freeze all layers except the last N encoder layers."""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers except the last N
        total_layers = len(self.bert.encoder.layer)
        layers_to_freeze = total_layers - num_layers_to_unfreeze

        for i in range(layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print(
            f"✅ Froze {layers_to_freeze} layers, unfroze last {num_layers_to_unfreeze} layers"
        )

        # Print trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with MLM head."""
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output (last hidden state)
        sequence_output = outputs.last_hidden_state

        # Apply MLM head
        prediction_scores = self.mlm_head(sequence_output)

        loss = None
        if labels is not None:
            # Only calculate loss on masked tokens (labels != -100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs.last_hidden_state,
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Main fine-tuning script."""

    # ========== Configuration ==========
    MODEL_NAME = "lhallee/CodonBERT"
    DATA_PATH = Path(__file__).parent.parent / "data" / "oligos.fasta"
    OUTPUT_DIR = Path(__file__).parent.parent / "models" / "finetuned_codonbert"

    # Training hyperparameters
    BATCH_SIZE = 4  # Reduced for stability
    LEARNING_RATE = 5e-6  # Lower learning rate to prevent NaN
    NUM_EPOCHS = 2  # Fewer epochs for testing
    MAX_LENGTH = 512
    MLM_PROBABILITY = 0.15
    NUM_LAYERS_TO_UNFREEZE = 3
    TRAIN_SPLIT = 0.8

    # Device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
    "mps" if torch.backends.mps.is_available() #M4 GPU
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
    print(f"Using device: {device}")

    # ========== Load Data ==========
    print("\n📊 Loading data...")
    from Bio import SeqIO
    import re

    def clean_sequence(seq):
        """Clean sequence by replacing IUPAC ambiguous bases with standard bases."""
        mapping = {
            "R": "A",
            "Y": "C",
            "D": "A",
            "H": "A",
            "N": "A",
            "B": "A",
            "V": "A",
            "W": "A",
            "S": "A",
            "M": "A",
            "K": "A",
        }

        # Expand patterns like (N20)
        def repl(match):
            base = match.group(1)
            num = int(match.group(2))
            base_clean = mapping.get(base, base) if base in mapping else base
            return base_clean * num

        seq = re.sub(r"\(([A-Z])(\d+)\)", repl, seq)
        seq = re.sub(r"[\(\)\d]", "", seq)
        seq = "".join(mapping.get(b, b) for b in seq)
        return seq

    def format_sequence_for_codonbert(seq):
        """Convert DNA sequence to space-separated codons for CodonBERT."""
        # CodonBERT expects space-separated codons
        codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
        return " ".join(codons)

    sequences = []
    for record in SeqIO.parse(str(DATA_PATH), "fasta"):
        seq = str(record.seq).upper()
        # Clean the sequence
        seq = clean_sequence(seq)
        # Only use sequences with length divisible by 3 and containing only ATCG
        if len(seq) % 3 == 0 and len(seq) > 0 and all(c in "ATCG" for c in seq):
            formatted_seq = format_sequence_for_codonbert(seq)
            sequences.append(formatted_seq)

    print(
        f"Loaded {len(sequences)} sequences (after cleaning)"
    )  # Split into train/validation
    split_idx = int(len(sequences) * TRAIN_SPLIT)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    print(f"Train: {len(train_sequences)}, Validation: {len(val_sequences)}")

    # ========== Initialize Tokenizer and Model ==========
    print("\n🔧 Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CodonBERTWithMLMHead(
        model_name=MODEL_NAME,
        freeze_layers=True,
        num_layers_to_unfreeze=NUM_LAYERS_TO_UNFREEZE,
    )
    model.to(device)

    # ========== Create Datasets and Dataloaders ==========
    print("\n📦 Creating datasets...")
    train_dataset = DNASequenceDataset(
        train_sequences, tokenizer, MAX_LENGTH, MLM_PROBABILITY
    )
    val_dataset = DNASequenceDataset(
        val_sequences, tokenizer, MAX_LENGTH, MLM_PROBABILITY
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========== Setup Optimizer and Scheduler ==========
    print("\n⚙️ Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ========== Training Loop ==========
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"💾 New best model! Saving to {OUTPUT_DIR}")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                OUTPUT_DIR / "best_model.pt",
            )

            # Save tokenizer
            tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
