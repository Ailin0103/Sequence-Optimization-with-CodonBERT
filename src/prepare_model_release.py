"""Validate local release assets and place the reviewed model card alongside them.

Does not load weights, evaluate data, authenticate, or upload anything.
"""
import json
from pathlib import Path
import math
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / 'models/cambray_codonbert_regression/final_model'
    for filename in ('model.safetensors', 'config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'target_scaler.json'):
        if not (model / filename).is_file():
            raise FileNotFoundError(model / filename)
    config = json.loads((model / 'config.json').read_text())
    if config.get('problem_type') != 'regression' or len(config.get('id2label', {})) != 1:
        raise ValueError('Expected a single-output regression model')
    scaler = json.loads((model / 'target_scaler.json').read_text())
    if scaler['target_column'] != 'clean.lin.prot.mean' or not math.isfinite(scaler['target_mean']) or not math.isfinite(scaler['target_std']) or scaler['target_std'] <= 0:
        raise ValueError('Invalid target scaling metadata')
    shutil.copyfile(ROOT / 'release/README.md', model / 'README.md')
    print(f'Local assets ready: {model}. No upload performed.')


if __name__ == '__main__':
    main()
