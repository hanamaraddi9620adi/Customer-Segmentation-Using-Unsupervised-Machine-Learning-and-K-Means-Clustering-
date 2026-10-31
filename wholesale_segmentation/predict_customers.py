"""Assign new spending records with the fitted K-Means model."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

def predict(input_path, output_path):
    base=Path(__file__).resolve().parent
    source=Path(input_path).resolve(); target=Path(output_path).resolve()
    if source==target:
        raise ValueError('Choose an output file different from the input file.')
    if target.exists():
        raise ValueError('Output already exists. Choose a new output filename.')
    bundle=joblib.load(base/'results'/'segmentation_model.joblib')
    data=pd.read_csv(source)
    missing=set(bundle['features'])-set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data.empty:
        raise ValueError('Input contains no customers.')
    values=data[bundle['features']].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(values.to_numpy()).all() or (values<0).any().any():
        raise ValueError('Spending must be finite, nonnegative and complete.')
    if 'cluster' in data:
        raise ValueError('Input already has a cluster column.')
    if not hasattr(bundle['model'],'predict'):
        raise ValueError('Saved model does not support assigning new customers.')
    data['cluster']=bundle['model'].predict(bundle['preprocessing'].transform(values))
    data.to_csv(target,index=False)
    print(f'Assigned {len(data)} customers. Saved {target}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv');parser.add_argument('output_csv')
    args=parser.parse_args()
    predict(args.input_csv,args.output_csv)
