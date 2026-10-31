from pathlib import Path
import base64, shutil, json
import nbformat, markdown
from nbclient import NotebookClient
base=Path(__file__).resolve().parent
source=base/'run_project.py'
code=source.read_text(encoding='utf-8-sig')
code=code.replace("BASE=Path(__file__).resolve().parent", "BASE=Path.cwd()")
markers=[('1. Imports and dataset validation','q1,q3='),('2. Explore distributions and prepare features','rows=[]; labels={}'),('3. Compare 14 clustering candidates','# Fixed selection rule'),('4. Select the model and test robustness','assigned=df.copy()'),('5. Export assignments and fitted model','pca=PCA('),('6. Create visualizations and record results',None)]
nb=nbformat.v4.new_notebook();cells=[nbformat.v4.new_markdown_cell('# Customer Segmentation Using Unsupervised Learning\n\nExecuted project notebook. Run cells in order from this folder. All six spending variables enter clustering; Channel and Region do not. Read PROJECT_WALKTHROUGH.md for interpretation. Generated results are overwritten on rerun.')]
rest=code
for title,boundary in markers:
 if boundary:
  pos=rest.index(boundary);part,rest=rest[:pos],rest[pos:]
 else:part=rest
 cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(part)])
cells.extend([nbformat.v4.new_markdown_cell('## 7. Inspect results\nSilhouette is an internal separation measure, not classification accuracy. The PCA plot is only a projection.'),nbformat.v4.new_code_cell("from IPython.display import display, Image\ndisplay(scores.round(4))\ndisplay(med)\ndisplay(Image(filename=str(OUT/'segmentation_overview.png')))"),nbformat.v4.new_markdown_cell('## 8. Conclusions and limitations\nK-Means with two groups is selected. Separation is modest (silhouette 0.2903), while mean subsampling ARI is 0.9649. Raw-versus-log ARI is 0.0936, showing strong preprocessing sensitivity. These exploratory in-sample results are not proof of business value or future-data performance.')])
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(base)}}).execute()
nbformat.validate(nb);nbformat.write(nb,base/'Customer_Segmentation.ipynb')
text=(base/'ABSTRACT.md').read_text(encoding='utf-8-sig')+'\n\n'+(base/'PROJECT_WALKTHROUGH.md').read_text(encoding='utf-8-sig')
body=markdown.markdown(text,extensions=['tables','fenced_code'])
images=''
for file in ['segmentation_overview.png','spending_distributions.png']:
 data=base64.b64encode((base/'results'/file).read_bytes()).decode()
 images+=f'<figure><img alt="{file}" src="data:image/png;base64,{data}"></figure>'
css='body{font:17px/1.65 system-ui,sans-serif;color:#1c2940;background:#eef2f7;margin:0}main{max-width:1100px;margin:35px auto;padding:45px;background:white;border-radius:12px}h1,h2{line-height:1.2;color:#173d65}h1{font-size:34px}h2{margin-top:42px}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}td,th{padding:10px;border-bottom:1px solid #d9e2ed;text-align:left}th{background:#edf4fa}pre{padding:18px;background:#f1f4f8;overflow:auto}img{width:100%;height:auto}figure{margin:24px 0}a{color:#1260a0}@media print{body{background:white}main{margin:0;padding:0}h2{break-after:avoid}figure{break-inside:avoid}}'
(base/'FINAL_REPORT.html').write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wholesale Customer Segmentation</title><style>{css}</style></head><body><main>{body}<h2>Result figures</h2>{images}</main></body></html>',encoding='utf-8')
print('Executed notebook and self-contained HTML report created.')
