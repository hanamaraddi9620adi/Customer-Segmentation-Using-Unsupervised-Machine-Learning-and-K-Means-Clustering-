# Customer Segmentation Using Unsupervised Learning

## What was completed
The first complete exploratory modeling workflow is implemented and executed on the uploaded Wholesale Customers file. The original file is unchanged. All 440 observations are retained. Other uploaded datasets are alternatives and were not combined with this study.

## Step 1: Define the problem
Discover groups of wholesale customers with similar annual purchasing patterns. One row is one customer observation. No target variable is supplied to the clustering algorithm. Cluster numbers are arbitrary identifiers, not ranks or known customer classes.

## Step 2: Select and understand the dataset
The local file has 440 rows and eight columns, consistent with the UCI Wholesale Customers description. Six columns represent spending: Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen. Channel and Region are categorical metadata retained only for interpretation. Monetary units are unspecified; do not label the figures as rupees or dollars. The local file was not independently verified byte-for-byte against the publisher's download.

## Step 3: Validate quality
Rechecked missing values, exact duplicate rows, finite numbers and nonnegative spending. All checks passed. No imputation, duplicate removal or row deletion was needed. An IQR rule flags 108 customers with at least one unusually high/low spending value. This is a statistical review flag, not evidence of corruption. Deleting all flagged rows would remove almost a quarter of the sample, so all were retained. See results/feature_audit.csv and spending_distributions.png.

## Step 4: Prepare features
Use only the six spending columns. Apply log(1 + spending) to reduce skew, then StandardScaler to center each transformed feature and give it unit variance. This makes clustering sensitive to proportional spending differences and gives each category comparable influence. Source row numbers, Channel, Region and outlier flags are excluded from training. The source_row output is a CSV line reference including the header, not a real customer ID.

## Step 5: Train candidate models
Fit K-Means and Ward agglomerative clustering for k = 2 through 8, producing 14 candidates on exactly the same prepared features. K-Means uses 30 initializations and random seed 42. Ward supplies a hierarchical comparison. This is a bounded comparison, not a claim that every clustering algorithm was tested.

## Step 6: Evaluate without labels
Calculate silhouette (higher is better), Davies-Bouldin (lower is better), and Calinski-Harabasz (higher is better). Record K-Means inertia and cluster sizes. The prespecified selection rule chooses the highest silhouette among candidates whose smallest cluster contains at least 5% of observations (22 customers). That minimum is a project choice, not a universal rule. Accuracy, precision and recall do not apply because there are no ground-truth cluster labels.

Selected result: K-Means, k = 2, silhouette = 0.2903, Davies-Bouldin = 1.3515, Calinski-Harabasz = 189.05. Ward with two clusters has silhouette 0.2585. The selected model has the best silhouette among the tested candidates but does not dominate every metric. A silhouette of 0.29 indicates modest separation with overlap, not perfectly distinct groups.

## Step 7: Check robustness
Refit preprocessing and the selected model on 20 random subsets, each containing 80% of customers. Compare each subset's assignments with the full-data assignments on those same rows using Adjusted Rand Index (ARI). Mean ARI = 0.9649; minimum = 0.8259; maximum = 1.0. This supports stability under this subsampling procedure. It is not independent predictive validation.

Also fit the selected algorithm and k on raw standardized spending without the log transform. ARI against the main solution is only 0.0936. The answer depends substantially on the definition of similarity. The log-based solution is appropriate for studying relative purchasing patterns; raw spending places more emphasis on large absolute expenditures. This sensitivity must be discussed in the project presentation. Scores from different feature spaces are not used to rank preprocessing alternatives.

## Step 8: Interpret the groups
These are descriptive working names based on median spending, not verified business identities.

| Cluster | Customers | Working description | Fresh | Milk | Grocery | Frozen | Detergents/Paper | Delicassen |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 252 | Higher fresh/frozen, lower grocery and detergents | 10,347.5 | 1,871.0 | 2,406.0 | 2,225.0 | 327.5 | 749.5 |
| 1 | 188 | Higher grocery/milk/detergents | 5,406.5 | 7,690.5 | 11,527.0 | 1,061.5 | 4,495.0 | 1,388.5 |

Possible applications include differentiated product bundles and inventory planning. These are hypotheses requiring business validation, not demonstrated commercial improvements. Channel and Region cross-tabulations are exported for post-model interpretation and did not influence model selection.

## Step 9: Visualize
The overview shows model comparison, cluster sizes, median spending profiles and a two-dimensional PCA projection. PCA retains 71.27% of variance in this display. Clustering was trained on all six prepared features, not the two plotted components. The projection loses information and must not be treated as proof of separation.

## Step 10: Save reproducible outputs
- run_project.py: executable workflow with fixed seed and explicit selection rule.
- requirements.txt: package versions used in this run (Python 3.14.5).
- results/customer_segments.csv: source rows, original values, cluster assignment and review flag.
- results/prepared_features.csv: six transformed model inputs in original row order.
- results/model_comparison.csv: all 14 candidate results.
- results/cluster_medians.csv: profiles in original monetary units.
- results/stability.csv: all 20 subsampling results.
- results/metrics.json: selected results, versions and source SHA-256 checksum.
- results/segmentation_model.joblib: fitted preprocessing and model; load only trusted model files.
- results/segmentation_overview.png and spending_distributions.png: charts.

To reproduce from a Python environment with the listed packages:

```powershell
cd E:\Corizo\wholesale_segmentation
python run_project.py
```

For a fresh environment, install requirements first with `python -m pip install -r requirements.txt`. Re-running overwrites generated results in this project folder. To assign new customers using the saved K-Means model, select the saved feature order, validate finite nonnegative values, call the saved preprocessing.transform, then model.predict. Do not refit the scaler on incoming customers.

## What remains before final submission
Review this dataset choice and project abstract with the team; adapt names and institution details; prepare the requested presentation or submission format. The computational exploratory workflow is complete. Deployment, independent future-data validation and business impact testing have not been performed. Because the objective is descriptive segmentation of this fixed dataset, all rows were used for model selection and final fitting. Report the scores as exploratory in-sample results, not held-out performance.

## References
- UCI Wholesale Customers dataset: https://archive.ics.uci.edu/dataset/292/wholesale+customers (DOI: 10.24432/C5030X). Dataset description and units.
- scikit-learn clustering documentation: https://scikit-learn.org/stable/modules/clustering.html . Algorithms and internal evaluation measures.

## Explain the project in a viva

**What makes this unsupervised?** The algorithm sees six spending measurements but no desired class or outcome. It discovers assignments by similarity. Channel and Region are used only after training.

**How does K-Means work?** It initializes k centers, assigns each observation to its closest center, updates centers to the mean of assigned observations, and repeats until convergence. Here distances are calculated in standardized log-spending space.

**Why use a log transform and scaling?** The log transform compresses very large purchases. Standardization prevents naturally larger-scale categories from dominating distance. The choice changes what similarity means, as our sensitivity analysis shows.

**Why two clusters?** Two had the highest silhouette in our defined search, subject to a minimum group size. This is evidence for a useful two-group summary, not proof of two true customer populations.

**What is silhouette?** For each observation, it compares average distance within its assigned cluster with average distance to the nearest alternative cluster: s = (b - a) / max(a, b). Values closer to 1 indicate better separation; around 0 indicates overlap; negative values suggest an observation might fit another cluster better. The reported score is the mean across customers.

**What is ARI?** Adjusted Rand Index compares two sets of assignments while accounting for chance agreement. It ignores arbitrary changes in cluster numbering. Here it measures resampling stability, not accuracy against real customer labels.

**Why not a conventional train/test accuracy score?** This is exploratory segmentation of the available customer sample. There is no known correct label. Future-data evaluation is still needed before operational deployment; resampling does not substitute for that.

**Why retain outliers?** Unusual spending can be a real business signal. We used transformation rather than automatically deleting 108 observations. More context would be needed to distinguish data errors from valuable large customers.

**What does PCA do here?** It gives a two-dimensional picture of six-dimensional data. It is not the clustering model, and the picture omits roughly 29% of transformed variance.

## Conclusion
The completed study produces an interpretable two-group summary of purchasing patterns. One group has higher fresh/frozen median spending; the other has higher grocery/milk/detergents spending. The result is stable under the tested subsampling procedure, but overlapping groups and sensitivity to preprocessing mean it should support exploration rather than automatic high-stakes decisions. Business experiments could test whether tailored bundles or stocking policies based on these profiles offer value. No improvement in revenue, retention or inventory efficiency was measured in this project.
