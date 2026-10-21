# Result artifacts

- `model_comparison.csv`: previously reported held-out test metrics on the original expression scale; unchanged.
- `training_history.csv`: authentic training/validation loss observations extracted from the final run's `checkpoint-36342/trainer_state.json`.
- `figures/codonbert_training_loss.png`: loss on standardized targets at actual logged epochs.

Historical screening summaries and their script are retained in `archive/`. Raw data, large prediction files, and checkpoints remain outside Git.
