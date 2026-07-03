# Iris Species Report

A beginner-friendly classification project on the classic **Iris dataset**, comparing Logistic Regression and a Decision Tree, with an Explainable AI (XAI) section that shows *why* the models make the predictions they do.

## What this script does

1. **Loads and explores** the Iris dataset (`Iris.csv`) — shape, info, summary stats, and correlation of each feature with species.
2. **Visualizes the data** — correlation heatmap, petal length vs. width scatter plot, and class distribution pie chart.
3. **Preprocesses the data** — encodes species labels, splits into train/test sets, scales features with `StandardScaler`, and balances the training set with `SMOTE` (kept in as good practice, even though Iris is already balanced).
4. **Trains two models**:
   - Logistic Regression
   - Decision Tree (max depth 4)
5. **Evaluates both models** — accuracy, recall, F1 score, MSE, classification report, and a model comparison bar chart.
6. **Shows a confusion matrix** for whichever model performed best.
7. **Explains the models (XAI)** using three techniques:
   - **Decision Tree Feature Importance** — how much each feature helped the tree split classes cleanly.
   - **Logistic Regression Coefficients** — how much each feature pushes predictions toward/away from each species.
   - **SHAP (SHapley Additive exPlanations)** — how much each feature contributed to individual predictions, summarized across the test set.

## Requirements

Install the required packages before running:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap
```

## How to run

1. Place `Iris.csv` in the same folder as the script.
2. Run the script top to bottom, in order (important if using a notebook — cells depend on earlier ones being run first):

```bash
python iris_species_report.py
```

Or open it as a Jupyter notebook (`# %%` markers denote notebook cells) and run all cells sequentially.

## Output files generated

| File | Description |
|---|---|
| `heatmap.png` | Correlation heatmap of features vs. species |
| `scatter_plot.png` | Petal length vs. petal width, colored by species |
| `pie_chart.png` | Class distribution of species |
| `model_comparison.png` | Accuracy comparison between models |
| `confusion_matrix.png` | Confusion matrix for the best-performing model |
| `xai_dt_importance.png` | Decision Tree feature importance |
| `xai_lr_coef.png` | Logistic Regression coefficient heatmap |
| `xai_shap_summary.png` | SHAP summary plot of feature contributions |

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains 150 samples across 3 species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*), with 4 features: sepal length, sepal width, petal length, and petal width (all in cm).

## Notes

- SMOTE is applied for consistency with other reports in this series, even though Iris is already perfectly balanced (50/50/50) and SMOTE has little effect here.
- Petal measurements typically prove far more predictive of species than sepal measurements — this is reflected consistently across all three XAI techniques.
## AUTHOR
ROHAN TIWARI , RESEARCHER ,CO- DSAI , TIET-UQ CENTER 
