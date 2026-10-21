"""Rebuild compact expression-scale reference statistics using training rows only."""
import ast
import json
from pathlib import Path
import math
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/train_codonbert_regression.py'
TARGET = 'clean.lin.prot.mean'
SEQUENCE = 'gs.sequence'


def main():
    # Reuse the established cleaning function without importing training libraries.
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'codon_tokenize')
    namespace = {'pd': pd, 'MAX_LENGTH': 64}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    frame = pd.read_csv(ROOT / 'data/Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv', usecols=[SEQUENCE, TARGET])
    frame = frame.dropna(subset=[SEQUENCE, TARGET]).copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors='coerce')
    frame = frame.dropna(subset=[TARGET])
    frame['formatted_seq'] = frame[SEQUENCE].apply(namespace['codon_tokenize'])
    frame = frame.dropna(subset=['formatted_seq']).reset_index(drop=True)
    train, _ = train_test_split(frame, test_size=.20, random_state=42)
    values = train[TARGET]
    scaler = json.loads((ROOT / 'models/cambray_codonbert_regression/final_model/target_scaler.json').read_text())
    assert math.isclose(values.mean(), scaler['target_mean'], rel_tol=1e-10)
    assert math.isclose(values.std(), scaler['target_std'], rel_tol=1e-10)
    reference = {'target_column': TARGET, 'reference_population': 'training split only',
                 'split_seed': 42, 'count': len(values), 'minimum': float(values.min()),
                 'p05': float(values.quantile(.05)), 'median': float(values.median()),
                 'p95': float(values.quantile(.95)), 'maximum': float(values.max()),
                 'mean': float(values.mean()), 'sample_std': float(values.std()),
                 'note': 'Observed training distribution, not hard output bounds or a prediction interval.'}
    (ROOT / 'demo/expression_reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps(reference, indent=2))


if __name__ == '__main__':
    main()
