from pathlib import Path
import json,joblib,numpy as np,pandas as pd,shutil
from sklearn.decomposition import PCA
base=Path(__file__).resolve().parent
p=base/'wholesale_segmentation/results'; out=base/'public'
b=joblib.load(p/'segmentation_model.joblib');d=pd.read_csv(p/'customer_segments.csv');z=b['preprocessing'].transform(d[b['features']]);pca=PCA(n_components=2).fit(z);xy=pca.transform(z)
model={'features':b['features'],'mean':b['preprocessing']['scale'].mean_.tolist(),'scale':b['preprocessing']['scale'].scale_.tolist(),'centers':b['model'].cluster_centers_.tolist()}
rows=[]
for i,r in d.iterrows():
 row={k:int(r[k]) for k in ['source_row','cluster','Channel','Region']+b['features']};row['x']=float(xy[i,0]);row['y']=float(xy[i,1]);rows.append(row)
(out/'data.json').write_text(json.dumps({'model':model,'customers':rows,'metrics':json.loads((p/'metrics.json').read_text()),'comparison':pd.read_csv(p/'model_comparison.csv').replace({np.nan:None}).to_dict('records')}))
for src,name in [(base/'wholesale_segmentation/FINAL_REPORT.html','report.html'),(base/'wholesale_segmentation/ABSTRACT.md','abstract.md'),(p/'customer_segments.csv','customer_segments.csv')]:shutil.copyfile(src,out/name)
print('Exported real fitted model and 440 customers')
