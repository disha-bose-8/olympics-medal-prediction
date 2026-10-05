# Olympics Medal Prediction: Reproduction Study

Mini project (Reproduce and Extend). This repository reproduces the results of the reference paper on predicting Summer Olympic medal counts from economic and demographic data, and adds one further model (Gradient Boosting).

- **Team:** `Deepika K (PES2UG24CS148)`, `Disha Bose (PES2UG24CS160)`
- **Course:** Machine Learning, PES University
- **Reference:** `2020 Summer Olympic Games Predictions`
- **Status:** Part 1 (reproduction) complete. Part 2 (additional model, Gradient Boosting): done.

## Problem

For each country at each Summer Olympics, predict:

1. **Classification:** will the country win at least one medal? (metric: accuracy)
2. **Regression:** how many medals will it win? (metric: RMSE)

Following the reference, models are trained on the 1988-2008 Games, validated on 2012, and the final model is tested on 2016.

## Dataset

| Item | Detail |
|---|---|
| Olympic data | Summer Games 1988-2016, medals and athletes per country (Kaggle: *120 years of Olympic history*, `athlete_events.csv`) |
| Economic/demographic data | World Bank WDI: GDP, GDP per capita, GDP growth, population, population growth, land area (via `wbgapi`) |
| Processed file | `data/processed/olympics_merged.csv` (committed, so the raw data is **not** needed to run the models) |
| Size | 197 countries in 2012 and 197 in 2016; rows with missing values are dropped, as in the reference |
| Target | `Medals` (regression) and `Medals > 0` (classification) |
| Features | The reference's 13 features: `Year, GDP, GDP_Per_Capita, GDP_Growth, GDP_pct_World, Pop, Pop_pct_World, Pop_Growth, Area, Athletes, Athletes_pct, Medals_Last_Games, Total_Medals_Year` |

## Repository structure

```
.
├── data/processed/olympics_merged.csv   # final modelling dataset
├── src/
│   ├── config.py                        # paths, train/val/test years, feature list
│   ├── download_wdi.py                  # (optional) fetch World Bank indicators
│   ├── data_prep.py                     # (optional) rebuild the merged dataset from raw files
│   ├── 02_run_reference_models.py       # Part 1: runs all reference models, writes comparison tables
│   ├── gradient_boosting.py             # Part 2: Gradient Boosting, compared with Part 1
│   ├── metrics.py                       # RMSE, MAE, top-10 RMSE, selection score
│   └── demo.py                          # Streamlit demo (both models)
├── results/
│   ├── reference_*.csv                  # values reported in the reference
│   ├── compare_classification_2012.csv  # Reported vs Your accuracy
│   ├── compare_regression_2012.csv      # Reported vs Your RMSE
│   ├── compare_final_2016.csv           # Reported vs Your final 2016 RMSE
│   ├── final_2016_predictions.csv       # Part 1 per-country 2016 predictions
│   └── gradient_boosting_*.csv          # Part 2 results and per-country 2016 predictions
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11 (other 3.10+ versions should also work).

```bash
git clone <repo-url>
cd olympics-medal-prediction
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## How to run

From the repository root:

```bash
python src/02_run_reference_models.py
python src/gradient_boosting.py
streamlit run src/demo.py
```

The first command takes a few minutes (hyperparameter searches). It prints the 2012 classification accuracies, the 2012 regression RMSEs and the final 2016 RMSE, and writes the three `compare_*.csv` tables and `final_2016_predictions.csv` to `results/`. Run it before `gradient_boosting.py`, which reads `final_2016_predictions.csv` for the Part 1 comparison and writes the `gradient_boosting_*.csv` files. The last command opens the demo, where you can switch between the two models.

### Rebuilding the dataset (optional)

Only needed if you want to regenerate `olympics_merged.csv` from scratch.

1. Download `athlete_events.csv` from Kaggle and place it in `data/raw/`.
2. `python src/download_wdi.py` (writes `data/raw/wdi_selected.csv`)
3. `python src/data_prep.py` (writes `data/processed/olympics_merged.csv`)

`data/raw/` is git-ignored.

## Approach

- **Preprocessing:** all models except Random Forest are wrapped in a `StandardScaler` pipeline, because features differ by orders of magnitude (e.g. `Pop` ~1e9 vs `GDP_Growth` ~1).
- **Classification models:** Logistic Regression, SVM (polynomial kernel, degree 1), Gaussian Naive Bayes, MLP, Random Forest. Hyperparameters tuned with 3-fold CV on the training years only.
- **Regression models:** Linear, Ridge, Lasso, SVR, Poisson, Random Forest, and a weighted Linear Regression. Trained on all training rows, evaluated on raw predictions.
- **Final model (2016):** Gaussian Naive Bayes decides whether a country wins any medal; Ridge regression predicts the count for those that do. Trained on 1988-2012, tested on 2016.
- **Part 2 (Gradient Boosting):** same two-stage structure, same features, split, scaler and metrics. A Gaussian NB gate decides medal or no medal; a Gradient Boosting regressor, trained only on medal-winning countries, predicts the count. Hyperparameters come from a random search (40 candidates, 3-fold CV, 1988-2008 only). A variant that predicts the change from `Medals_Last_Games` was compared with the plain variant on 2012 only, and the better one (change target) was run once on 2016. Predictions are clipped at 0 and not rounded.
- **Reproducibility:** `random_state=42` throughout.

## Results (Reported vs Ours)

**Classification, 2012 accuracy**

| Model | Reported | Ours |
|---|---|---|
| Logistic Regression | 0.8832 | 0.8782 |
| SVM | 0.8629 | 0.8832 |
| Gaussian Naive Bayes | 0.7614 | 0.8680 |
| MLP | 0.8782 | 0.8832 |
| Random Forest | 0.8731 | 0.8832 |

**Regression, 2012 RMSE**

| Model | Reported | Ours |
|---|---|---|
| Linear Regression | 3.1115 | 3.1115 |
| Ridge | 3.1582 | 3.0989 |
| Lasso | 3.0386 | 3.0510 |
| SVR | 2.8156 | 2.6936 |
| Poisson | 5.8460 | 5.0104 |
| Random Forest | 2.8869 | 2.3446 |
| Weighted Linear Regression | 6.6632 | 3.8508 |

**Final 2016 (Gaussian NB -> Ridge) RMSE:** reported 4.8243, ours 3.5673.

"Reported" is the set of values stored in `results/reference_*.csv`. The paper's own Table 3 lists somewhat different figures (for example Gaussian NB 0.891, and Ridge 2.27 with a classifier gate).

### Differences from the reference

Linear Regression matches the reported value exactly, which suggests our dataset and features match the reference's. Several results differ clearly: Gaussian Naive Bayes (0.868 vs 0.761), Weighted Linear Regression (3.85 vs 6.66), Poisson, Random Forest and SVR, and the final 2016 RMSE. Likely reasons: our feature scaling (the reference code does not scale), unreported preprocessing and hyperparameter search ranges, random seeds and library versions. The weighted model uses the weighting from the reference code, `exp(-(y-50)^2/1000)`, while the paper's Eq. 4 states a different scale. In the reference code the final 2016 scoring appears to compare its 2016 predictions with the 2012 labels, which would inflate its error; we score against the true 2016 labels (this is our reading of the code, which we have not rerun). We report our lower final 2016 RMSE as a reproduction difference, not as an improvement.

## Part 2: Gradient Boosting results

**2012 classification accuracy:** Gradient Boosting 0.8629 (default and tuned), against 0.8680 to 0.8832 for the Part 1 classifiers.

**2012 regression RMSE (plain, no gate):** Gradient Boosting 3.0056. It beats Linear (3.1115), Ridge (3.0989), Lasso (3.0510), Weighted (3.8508) and Poisson (5.0104), and is behind Random Forest (2.3446) and SVR (2.6936).

**Final 2016 test (Gaussian NB gate, trained on 1988-2012)**

| Model | RMSE | MAE | Top-10 RMSE | Selection score |
|---|---|---|---|---|
| Part 1: GNB + Ridge (all training rows) | 3.567 | 1.502 | 10.021 | 6.073 |
| GNB + Ridge (medal-winning rows only) | 3.700 | 1.547 | 10.073 | 6.218 |
| Part 2: GNB + Gradient Boosting | 3.521 | 1.423 | 12.455 | 6.634 |

Selection score = RMSE + 0.25 x top-10 RMSE (lower is better). Gradient Boosting has slightly lower average error on 2016, but is worse on the largest countries and on the selection score, mainly because it over-predicts Russia (81.5 predicted, 56 actual). With one test year of 197 countries, we do not claim either model is better overall. Full tables are in `results/gradient_boosting_*.csv` and in the report.

## Team contributions

| Member | Contribution |
|---|---|
| `Deepika K` | `Part 1 implementing classification and regression models according to the given paper` |
| `Disha Bose` | `Part 2. Implementing a new model and comparing with the previous chosen ones` |

## Library versions

Results were produced with the versions in `requirements.txt` (pandas, numpy, scikit-learn). Different scikit-learn versions can change results slightly.