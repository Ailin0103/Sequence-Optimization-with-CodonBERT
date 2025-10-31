# CodonBERT Fine-tuning Summary

## Overview

Successfully created a fine-tuning pipeline for CodonBERT on DNA sequences with masked language modeling (MLM).

## Key Files Created

### 1. `finetune_codonbert.py`

Main training script that:

- **Loads and cleans DNA sequences** from FASTA files
- **Formats sequences** as space-separated codons (required by CodonBERT)
- **Freezes layers**: Only trains the last 3 transformer layers (25.31% of parameters)
- **Uses MLM objective**: Randomly masks 15% of tokens and trains model to predict them
- **Saves best model** based on validation loss

**Key Features:**

- Automatic sequence cleaning (handles IUPAC ambiguous bases)
- Proper codon formatting for CodonBERT tokenizer
- Layer freezing for efficient fine-tuning
- Learning rate: 5e-6 (low to prevent instability)
- Batch size: 4 (for stability on CPU)

### 2. `test_finetuned_model.py`

Testing script that:

- Loads the fine-tuned model checkpoint
- Generates embeddings for test sequences
- Shows model predictions

## Training Results

**Dataset:** 8 sequences from `oligos.fasta` (after cleaning)

- Train: 6 sequences
- Validation: 2 sequences

**Performance:**

- Epoch 1: Train Loss = 4.08, Val Loss = 3.85
- Epoch 2: Train Loss = 4.14, Val Loss = 3.75 ✅ (best)

**Model saved to:** `models/finetuned_codonbert/`

## Important Lessons Learned

### 1. CodonBERT Requires Space-Separated Codons

CodonBERT doesn't use character-level tokenization. Sequences must be formatted as:

```
Input:  ATGAAACCCGGG
Format: ATG AAA CCC GGG
```

### 2. Clean Sequences Required

- Remove IUPAC ambiguous bases (N, Y, R, etc.)
- Only use A, T, C, G
- Ensure length is divisible by 3

### 3. Layer Freezing

- Total parameters: 86,540,613
- Trainable (last 3 layers): 21,907,269 (25.31%)
- Significantly reduces training time and prevents overfitting

## Usage Example

### Fine-tune the model

```python
python src/sequence_optimization_with_codonbert/finetune_codonbert.py
```

### Test the fine-tuned model

```python
python src/sequence_optimization_with_codonbert/test_finetuned_model.py
```

### Use in notebook

```python
from transformers import AutoTokenizer
import torch
from pathlib import Path

# Load model
model_dir = Path("models/finetuned_codonbert")
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
checkpoint = torch.load(model_dir / "best_model.pt")

# Format sequence
def format_sequence(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq)-len(seq)%3, 3)]
    return " ".join(codons)

# Get embeddings
seq = format_sequence("ATGAAACCCGGG")
inputs = tokenizer(seq, return_tensors="pt")
embeddings = model(**inputs)["hidden_states"]
```

## Next Steps

1. **Increase dataset size**: More sequences will improve model performance
2. **Tune hyperparameters**: Experiment with learning rate, batch size, epochs
3. **Try different freezing strategies**: Unfreeze different numbers of layers
4. **Add task-specific heads**: Add classification/regression heads for downstream tasks
5. **Evaluate on specific tasks**: Test on codon optimization, gene expression prediction, etc.
