"""
Test the fine-tuned CodonBERT model by generating embeddings and predictions.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


class CodonBERTWithMLMHead(nn.Module):
    """CodonBERT model with MLM head for fine-tuning."""

    def __init__(
        self,
        model_name="lhallee/CodonBERT",
        freeze_layers=False,
        num_layers_to_unfreeze=3,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.mlm_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.mlm_head.bias is not None:
            self.mlm_head.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs.last_hidden_state,
        }


def format_sequence_for_codonbert(seq):
    """Convert DNA sequence to space-separated codons."""
    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)


def main():
    # Paths
    MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "finetuned_codonbert"
    CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # Load model
    print("🔧 Loading fine-tuned model...")
    model = CodonBERTWithMLMHead(model_name="lhallee/CodonBERT")

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}\n")

    # Test sequences
    test_sequences = [
        "ATGAAACCCGGGTTT",  # Simple sequence
        "ATGGCTAGCTAGCTA",  # Another test
    ]

    print("🧪 Testing on sample sequences:\n")

    for i, seq in enumerate(test_sequences, 1):
        print(f"Sequence {i}: {seq}")

        # Format for CodonBERT
        formatted_seq = format_sequence_for_codonbert(seq)
        print(f"Formatted: {formatted_seq}")

        # Tokenize
        inputs = tokenizer(
            formatted_seq,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Get predictions
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )

        # Get embeddings
        embeddings = outputs["hidden_states"]  # [batch, seq_len, hidden_dim]

        print(f"Embedding shape: {embeddings.shape}")
        print(f"Mean embedding norm: {embeddings.norm(dim=-1).mean():.4f}")

        # Get predicted tokens (for masked language modeling)
        logits = outputs["logits"]
        predicted_token_ids = logits.argmax(dim=-1)

        # Decode predictions
        predicted_tokens = tokenizer.batch_decode(
            predicted_token_ids, skip_special_tokens=True
        )
        print(f"Predicted sequence: {predicted_tokens[0]}\n")

    print("✅ Testing complete!")


if __name__ == "__main__":
    main()
