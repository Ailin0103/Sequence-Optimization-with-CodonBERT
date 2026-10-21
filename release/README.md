---
library_name: transformers
base_model: lhallee/CodonBERT
tags:
- biology
- regression
- protein-expression
---

# CodonBERT for Cambray protein-expression regression

Pretrained `lhallee/CodonBERT`, specialized by full supervised fine-tuning on
synthetic E. coli sequences for continuous protein-expression regression.
This card documents the existing fine-tuned model.

## Training data

Cambray et al. (2018), Supplementary Data 15: 242,269 usable sequences;
193,815 training, 24,227 validation, and 24,227 test examples. Input:
`gs.sequence`; target: `clean.lin.prot.mean`. Split seed 42, 80/10/10.
Raw training data are not redistributed with the model.

DNA is uppercased, stripped, converted to RNA, and split into non-overlapping
codons. Invalid bases and non-frame-aligned sequences are excluded. Maximum
length is 64 tokens including special tokens. Training uses learning rate 1e-5,
three epochs, batch size 16, and training-set target standardization.

## Evaluation

Previously reported held-out test metrics, on the original expression scale:

| RMSE | R² | Pearson | Spearman |
|---:|---:|---:|---:|
| 14.0871 | 0.7186 | 0.8554 | 0.8557 |

No new experiments or test evaluations were performed to prepare this release.

## Intended use and limitations

Research and exploratory expression prediction on related coding-sequence
distributions. Training covers a specific synthetic E. coli library. External
generalization has not been established. Predictions are not experimentally
validated expression measurements for arbitrary organisms or sequence distributions.
Confirm upstream model and dataset license terms before redistribution; this
card does not assert a new license for upstream materials.

## Inference

```python
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

model_dir = Path('models/cambray_codonbert_regression/final_model')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir).eval()
scaler = json.loads((model_dir / 'target_scaler.json').read_text())
sequence = 'ATGGCC'  # Replace with an appropriate related coding sequence.
rna = sequence.upper().strip().replace('T', 'U')
if not rna or len(rna) % 3 or set(rna) - set('AUGC') or len(rna) // 3 > 62:
    raise ValueError('Expected a valid, frame-aligned coding sequence of <=62 codons')
formatted = ' '.join(rna[i:i+3] for i in range(0, len(rna), 3))
inputs = tokenizer(formatted, return_tensors='pt', truncation=False)
with torch.no_grad():
    standardized_prediction = model(**inputs).logits.item()
expression = standardized_prediction * scaler['target_std'] + scaler['target_mean']
print(expression)
```

Scaling is reversed as `expression = prediction_z * training_std + training_mean`.
The standard deviation uses pandas' sample convention (`ddof=1`). Actual values
are stored in `target_scaler.json`; never estimate them from inference examples.

## Reference

Cambray G, Guimaraes JC, Arkin AP. Evaluation of 244,000 synthetic sequences
reveals design principles to optimize translation in Escherichia coli.
Nature Biotechnology (2018).
