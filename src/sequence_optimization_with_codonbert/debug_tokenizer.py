"""
Debug script to check tokenizer and data issues
"""

import torch
from transformers import AutoTokenizer
from Bio import SeqIO
from pathlib import Path

# Load tokenizer
MODEL_NAME = "lhallee/CodonBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer info:")
print(f"Vocab size: {len(tokenizer)}")
print(f"Mask token: {tokenizer.mask_token} (id: {tokenizer.mask_token_id})")
print(f"Special tokens: {tokenizer.special_tokens_map}")

# Load a sequence
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "oligos.fasta"


def format_sequence_for_codonbert(seq):
    """Convert DNA sequence to space-separated codons for CodonBERT."""
    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)


sequences = []
for record in SeqIO.parse(str(DATA_PATH), "fasta"):
    seq = str(record.seq).upper()
    if len(seq) % 3 == 0:
        sequences.append(format_sequence_for_codonbert(seq))
        if len(sequences) >= 1:
            break

print(f"\nTest sequence (raw): {sequences[0][:100]}...")
print(f"Length: {len(sequences[0])}")

# Tokenize
encoding = tokenizer(
    sequences[0],
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512,
)

print(f"\nEncoded input_ids shape: {encoding['input_ids'].shape}")
print(f"First 20 tokens: {encoding['input_ids'][0][:20].tolist()}")
print(
    f"Token range: min={encoding['input_ids'].min()}, max={encoding['input_ids'].max()}"
)

# Try decoding
decoded = tokenizer.decode(encoding["input_ids"][0][:50])
print(f"\nDecoded (first 50 tokens): {decoded[:200]}")
