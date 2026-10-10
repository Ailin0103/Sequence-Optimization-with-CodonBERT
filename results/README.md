# Final Results

This directory contains compact summary results from the final Cambray protein-expression experiments.

- `model_comparison.csv`: held-out test regression performance for the 7-mer TF-IDF Ridge baseline and fine-tuned CodonBERT.
- `screening_comparison.csv`: favorable-expression screening performance. The favorable experimental range is defined as PNI 40-60, based on the region associated with near-maximal average cellular fitness in the Cambray dataset.

Prediction intervals were selected using the validation set with a minimum recall requirement of 80% and then evaluated once on the held-out test set.

Large prediction files, model checkpoints, and raw datasets are not stored in this repository.
