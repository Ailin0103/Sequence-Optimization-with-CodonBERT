"""Extract authentic Trainer history and plot losses at their logged epochs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    history = json.loads(args.state.read_text())['log_history']
    rows = [{'step': item['step'], 'epoch': item.get('epoch', ''),
             'training_loss': item.get('loss', ''), 'validation_loss': item.get('eval_loss', '')}
            for item in history if 'loss' in item or 'eval_loss' in item]
    if not any(row['training_loss'] != '' for row in rows) or not any(row['validation_loss'] != '' for row in rows):
        raise ValueError('Both training and validation loss history are required')
    with (ROOT / 'results/training_history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.2, 3.5), layout='constrained')
    for key, label in [('training_loss', 'Training'), ('validation_loss', 'Validation')]:
        points = [row for row in rows if row[key] != '']
        ax.plot([row['epoch'] for row in points], [row[key] for row in points],
                label=label, linewidth=1.3, marker='o' if key == 'validation_loss' else None)
    ax.set(xlabel='Epoch', ylabel='Mean squared error (standardized target)')
    ax.legend(frameon=False)
    destination = ROOT / 'results/figures/codonbert_training_loss.png'
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=300)
    print('Validation observations:', [(r['epoch'], r['validation_loss']) for r in rows if r['validation_loss'] != ''])


if __name__ == '__main__':
    main()
