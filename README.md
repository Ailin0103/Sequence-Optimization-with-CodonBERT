# Sequence Optimization with CodonBERT

Fine-tuning a pretrained codon-level language model to predict protein expression and prioritize candidate sequences for experimental validation.

## Overview

This project evaluates whether **CodonBERT** can learn sequence features associated with protein production in *E. coli* and whether improved prediction accuracy translates into more efficient downstream sequence screening.

The final workflow is:

```text
Coding sequence
      ↓
Protein-expression prediction
      ↓
CodonBERT vs. 7-mer TF-IDF baseline
      ↓
Screening for a fitness-favorable expression range
      ↓
Candidate prioritization for experimental validation
```

The final CodonBERT model was trained on more than 190,000 experimental sequences and substantially outperformed a full-data 7-mer TF-IDF Ridge baseline.

On the held-out test set, CodonBERT achieved:

- **R² = 0.719**
- **RMSE = 14.09**
- **Spearman = 0.856**

For downstream screening, CodonBERT achieved **1.78× enrichment** while reducing the experimental candidate pool by **54.5%** at approximately 81% recall.

---

## Data

The final supervised modeling experiments use **Supplementary Data 15** from:

> Cambray G, Guimaraes JC, Arkin AP.  
> *Evaluation of 244,000 synthetic sequences reveals design principles to optimize translation in Escherichia coli.*  
> Nature Biotechnology, 2018.

The dataset contains designed nucleotide sequences together with experimentally measured protein production, cellular growth, RNA abundance, and polysome phenotypes.

The final prediction task uses:

```text
Input:   gs.sequence
Target:  clean.lin.prot.mean
```

After filtering for valid sequences and non-missing protein-expression measurements:

```text
Total usable sequences: 242,269

Train:       193,815
Validation:   24,227
Test:         24,227
```

Raw datasets are not included in this repository.

After obtaining Supplementary Data 15 from the original study, place it at:

```text
data/Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv
```

---

## Models

### 7-mer TF-IDF + Ridge Baseline

A conventional sequence baseline was constructed by converting each nucleotide sequence into overlapping 7-mers, applying TF-IDF, and fitting Ridge regression.

```text
Sequence
   ↓
Overlapping 7-mers
   ↓
TF-IDF
   ↓
Ridge regression
   ↓
Protein expression
```

Final configuration:

```text
k = 7
Ridge alpha = 1.0
Vocabulary size = 16,345
```

### CodonBERT

DNA sequences are converted to RNA and tokenized into non-overlapping codons before being passed to the pretrained CodonBERT model.

```text
DNA sequence
     ↓
RNA conversion
     ↓
Codon tokenization
     ↓
Pretrained CodonBERT
     ↓
Regression head
     ↓
Protein expression
```

The final model uses full supervised fine-tuning:

```text
Learning rate:  1e-5
Epochs:         3
Batch size:     16
Training size:  193,815 sequences
```

The regression target is standardized using training-set statistics during optimization and transformed back to the original protein-expression scale for evaluation.

---

## Protein-Expression Prediction

Both models were evaluated on the same held-out test set.

| Model | RMSE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|
| 7-mer TF-IDF + Ridge | 17.87 | 0.547 | 0.740 | 0.745 |
| **Fine-tuned CodonBERT** | **14.09** | **0.719** | **0.855** | **0.856** |

Compared with the full-data k-mer baseline, CodonBERT:

- reduced RMSE by approximately **21%**
- increased R² from **0.547 to 0.719**
- increased Spearman correlation from **0.745 to 0.856**

These results indicate that supervised CodonBERT fine-tuning captures predictive sequence information beyond overlapping k-mer frequency features.

Summary results are stored in:

```text
results/model_comparison.csv
```

---

## Fitness-Favorable Expression Screening

The Cambray dataset also provides relative cellular growth under normal translation initiation conditions (`growth.aggregate.R`).

Analysis of measured protein production and cellular growth showed a non-monotonic relationship.

A quadratic fit estimated the population-level growth optimum at:

```text
PNI ≈ 45.7
```

and decile analysis showed a broad near-maximal fitness region at intermediate protein-production levels.

Based on these observations, this project defines:

```text
PNI 40–60
```

as an **operational fitness-favorable expression range** for downstream candidate screening.

This range is not a threshold defined by Cambray et al., and it does not imply that every sequence with PNI 40–60 has high cellular fitness. It represents a protein-expression region associated with near-maximal average growth in this dataset.

---

## Screening Strategy

The regression models are not retrained for this task.

Instead, the validation set is used to identify a predicted-expression interval that:

1. recovers at least **80%** of sequences whose measured PNI lies between 40 and 60; and
2. maximizes enrichment among selected candidates.

The resulting screening interval is then frozen and evaluated on the held-out test set.

Validation-selected intervals were:

```text
7-mer Ridge:  predicted PNI 33–65
CodonBERT:    predicted PNI 30–71
```

The wider prediction intervals account for regression uncertainty while maintaining high recall.

---

## Screening Results

| Model | Recall | Precision | Enrichment | Pool Reduction | Candidates Selected |
|---|---:|---:|---:|---:|---:|
| 7-mer TF-IDF + Ridge | 80.26% | 31.35% | 1.40× | 42.78% | 13,863 |
| **CodonBERT** | **80.83%** | **39.70%** | **1.78×** | **54.50%** | **11,024** |

The prevalence of favorable sequences in the unscreened test population was **22.35%**.

After CodonBERT screening, **39.70%** of selected candidates were truly within the favorable PNI range, corresponding to **1.78× enrichment**.

At approximately matched recall, CodonBERT:

```text
selected 2,839 fewer candidates
```

while recovering slightly more favorable sequences:

```text
7-mer Ridge:
    4,346 favorable sequences recovered
    13,863 sequences selected

CodonBERT:
    4,377 favorable sequences recovered
    11,024 sequences selected
```

This demonstrates that improved regression performance translates into more efficient candidate prioritization for downstream experimental validation.

Summary results are stored in:

```text
results/screening_comparison.csv
```

---

## Model Development

The final workflow was developed through several earlier experiments retained in `exploratory/`.

These included:

- pretrained CodonBERT embedding extraction
- PCA-based representation analysis
- frozen and partially frozen CodonBERT models
- masked-language-model fine-tuning experiments
- smaller 10,000-sequence supervised experiments
- comparison with RNA-based pretrained models

Early frozen and partially frozen representations showed limited task-specific predictive performance.

This motivated two important changes:

1. standardizing the continuous protein-expression target during training
2. performing full end-to-end supervised fine-tuning

Scaling the final approach to the complete Cambray dataset produced the performance reported above.

---

## Early Data Engineering Pipeline

Earlier stages of the project also explored scalable processing of public biological sequence data.

The workflow included:

```text
FASTA ingestion
      ↓
AWS S3 storage and streaming
      ↓
PostgreSQL / Amazon RDS
      ↓
CodonBERT embedding extraction
      ↓
Representation analysis
```

This infrastructure is retained under:

```text
pipeline/
```

and is separate from the final Cambray supervised benchmark.

Account-specific AWS/RDS configuration is not stored in the repository. A local configuration can be created from:

```text
pipeline/config.example.py
```

---

## Validation and Sanity Checks

Small-scale experiments were used to verify:

- CodonBERT checkpoint/tokenizer compatibility
- codon tokenization
- sequence preprocessing
- model forward passes
- training-pipeline functionality

A small *E. coli* task was used as a pipeline sanity check rather than as a final benchmark.

These utilities are retained under:

```text
validation/
```

---

## Repository Structure

```text
Sequence-Optimization-with-CodonBERT/
│
├── src/
│   ├── train_codonbert_regression.py
│   ├── train_kmer_baseline.py
│   └── evaluate_favorable_expression_screening.py
│
├── exploratory/
│   └── earlier modeling and representation experiments
│
├── pipeline/
│   └── AWS / PostgreSQL / FASTA processing utilities
│
├── validation/
│   └── tokenizer and model sanity checks
│
├── archive/
│   └── deprecated experiments
│
├── results/
│   ├── model_comparison.csv
│   └── screening_comparison.csv
│
├── data/
│   └── local raw data (not tracked by Git)
│
├── pyproject.toml
├── pixi.lock
└── README.md
```

---

## Installation

1. Install `pixi` package management tool:

    ``` bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

2. Install python environment using `pixi`:

    ``` bash
    pixi install
    ```


---

## Reproducing the Final Experiments

Place Supplementary Data 15 at:

```text
data/Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv
```

Train the full-data k-mer baseline:

```bash
pixi run python src/train_kmer_baseline.py
```

Fine-tune CodonBERT:

```bash
pixi run python src/train_codonbert_regression.py
```

Evaluate favorable-expression screening:

```bash
pixi run python src/evaluate_favorable_expression_screening.py
```

The screening rule is selected using the validation set and evaluated using the held-out test set.

---

## Limitations and Future Work

The current downstream screening strategy predicts **protein expression**, rather than cellular fitness directly.

The PNI 40–60 target is based on a population-level association between intermediate protein production and near-maximal average cellular growth. Individual sequences within the same protein-expression range may still differ substantially in fitness.

Therefore, selected sequences should be interpreted as **enriched candidates for experimental validation**, not as sequences guaranteed to have optimal cellular fitness.

A natural extension would jointly predict protein production and cellular fitness and prioritize candidates based on both phenotypes.

---

## Reference

Cambray G, Guimaraes JC, Arkin AP.  
*Evaluation of 244,000 synthetic sequences reveals design principles to optimize translation in Escherichia coli.*  
Nature Biotechnology, 2018.


