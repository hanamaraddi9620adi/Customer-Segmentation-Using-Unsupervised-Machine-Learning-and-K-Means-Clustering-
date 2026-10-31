from pathlib import Path
import tempfile, json, zipfile
import pandas as pd
import numpy as np
import nbformat
from predict_customers import predict
base=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
 t=Path(temp); raw=base.parent/'Wholesale customers data3.csv'
 predict(raw,t/'predicted.csv')
 a=pd.read_csv(t/'predicted.csv'); b=pd.read_csv(base/'results/customer_segments.csv')
 assert np.array_equal(a.cluster,b.cluster)
 invalid=pd.read_csv(raw).head(2); invalid.loc[0,'Milk']=-1; invalid.to_csv(t/'invalid.csv',index=False)
 try:predict(t/'invalid.csv',t/'bad.csv')
 except ValueError:pass
 else:raise AssertionError('Negative spending accepted')
 invalid=invalid.drop(columns='Fresh');invalid.to_csv(t/'missing.csv',index=False)
 try:predict(t/'missing.csv',t/'bad.csv')
 except ValueError:pass
 else:raise AssertionError('Missing required feature accepted')
nb=nbformat.read(base/'Customer_Segmentation.ipynb',as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
checks={'all_440_saved_predictions_match':True,'negative_spending_rejected':True,'missing_feature_rejected':True,'notebook_all_code_cells_executed_without_errors':True}
(base/'results/verification.json').write_text(json.dumps(checks,indent=2))
zip_path=base.parent/'Wholesale_Segmentation_Complete_Project.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
 archive.write(raw,raw.name)
 for p in base.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts:
   archive.write(p,str(Path(base.name)/p.relative_to(base)))
with zipfile.ZipFile(zip_path) as archive:assert archive.testzip() is None
print(json.dumps(checks,indent=2));print(zip_path)
