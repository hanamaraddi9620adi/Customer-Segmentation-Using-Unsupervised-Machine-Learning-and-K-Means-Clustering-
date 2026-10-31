# Start here

Project: Customer Segmentation Using Unsupervised Learning.

1. Open FINAL_REPORT.html in a browser for the complete illustrated report. It works offline.
2. Read ABSTRACT.md for the assignment abstract.
3. Open Customer_Segmentation.ipynb in VS Code or Jupyter for executed code and outputs. Select a Python environment with requirements.txt installed; run from this folder.
4. Run `python run_project.py` to reproduce all model results.
5. Inspect results/customer_segments.csv for assignments and results/model_comparison.csv for model selection.

## Setup
Tested with Python 3.14.5. From this folder run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_project.py
```

Notebook/report rebuild dependencies are listed separately in requirements-notebook.txt. Run `python build_submission.py` to re-execute the notebook and regenerate the HTML report after installing them. Notebook and script runs overwrite generated results. The source CSV must be in the parent folder; the ZIP preserves this layout.

## Assign new customers
The input CSV must include Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen with complete finite nonnegative spending in the same units and observation period as training. Extra identifying columns are preserved. Example:

```powershell
python predict_customers.py new_customers.csv new_customer_segments.csv
```

Choose a new output filename. The saved preprocessing is reused, never refitted on incoming rows. Cluster numbers are identifiers, not quality scores. The tool does not detect distribution drift and has not been validated on independent future customer data. Load only the trusted model supplied with this project.

## Result
K-Means with two groups: 252 and 188 customers. Silhouette 0.2903 means modest separation. Mean subsampling ARI 0.9649 indicates stability under the tested resampling procedure. Preprocessing sensitivity remains substantial. Consult the report for details.

## Dataset credit
UCI Wholesale Customers, DOI 10.24432/C5030X, https://archive.ics.uci.edu/dataset/292/wholesale+customers . Publisher lists CC BY 4.0. The supplied local data is included for reproducibility; no claim of independent collection is made.
