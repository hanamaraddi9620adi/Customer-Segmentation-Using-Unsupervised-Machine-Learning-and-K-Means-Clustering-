# Segment Lab — Customer Segmentation

An interactive frontend and reproducible unsupervised-learning study on 440 wholesale customers.

## Frontend
- Filterable PCA customer map with point details.
- Searchable, paginated customer table and filtered CSV export.
- Live predictions using the real fitted K-Means centroids, log transform and scaler.
- Spending presets, segment median profiles and model-comparison charts.
- Responsive layout and downloadable research report.

Predictions run locally in the browser. No credentials or Python server are needed. Browser inference is tested against all 440 original Python assignments.

## Run locally
Requires Node.js 20+; no npm dependencies.

```sh
npm test
npm run dev
```

Open http://127.0.0.1:4173 .

## Deployment
Vercel serves `public/` as a static site using vercel.json. Deploy using `vercel --prod` or connect this repository in Vercel. No environment variables are required.

## Research
The `wholesale_segmentation/` directory contains the executed notebook, abstract, report, Python pipeline, saved model and results. Install its requirements and run `python wholesale_segmentation/run_project.py` to reproduce. Then run `python export_web.py` to refresh the frontend model/data. Run `npm test` after exporting.

Selected model: K-Means, k=2. Cluster sizes: 252 and 188. Silhouette: 0.2903. Mean subsampling ARI: 0.9649. These are exploratory segments with modest separation; raw-vs-log ARI of 0.0936 demonstrates preprocessing sensitivity. No predictive accuracy or measured business improvement is claimed. Cluster descriptions refer to group medians, not a rule that every assigned customer spends more in those categories.

Source row numbers reference CSV lines, not personal customer IDs. Channel and Region are excluded from training. Monetary units are unspecified.

Dataset credit: UCI Wholesale Customers, https://archive.ics.uci.edu/dataset/292/wholesale+customers (DOI 10.24432/C5030X), CC BY 4.0 according to publisher. Original uploaded source included for reproducibility.
