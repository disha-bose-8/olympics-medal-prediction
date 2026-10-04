# Olympics Medal Prediction: Reproduction Study

Mini project (Reproduce and Extend). This repository reproduces the results of the reference paper on predicting Summer Olympic medal counts from economic and demographic data.

- **Team:** `Deepika K (PES2UG24CS148)`, `Disha Bose (PES2UG24CS160)`
- **Course:** Machine Learning, PES University
- **Reference:** `2020 Summer Olympic Games Predictions`
- **Status:** Part 1 (reproduction) complete. Part 2 (additional model): `<done / in progress / not included>`

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
| Features | `Year, GDP_Growth, GDP_Per_Capita, Pop, Pop_pct_World, Athletes, Athletes_pct, Medals_Last_Games, Total_Medals_Year` |

## Repository structure

```
.
├── data/processed/olympics_merged.csv   # final modelling dataset
├── src/
│   ├── config.py                        # paths, train/val/test years, feature list
│   ├── download_wdi.py                  # (optional) fetch World Bank indicators
│   ├── data_prep.py                     # (optional) rebuild the merged dataset from raw files
│   └── 02_run_reference_models.py       # Part 1: runs all reference models, writes comparison tables
├── results/
│   ├── reference_*.csv                  # values reported in the reference
│   ├── compare_classification_2012.csv  # Reported vs Your accuracy
│   ├── compare_regression_2012.csv      # Reported vs Your RMSE
│   ├── compare_final_2016.csv           # Reported vs Your final 2016 RMSE
│   └── final_2016_predictions.csv       # per-country 2016 predictions
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
```

This takes a few minutes (hyperparameter searches). It prints the 2012 classification accuracies, the 2012 regression RMSEs and the final 2016 RMSE, and writes the three `compare_*.csv` tables and `final_2016_predictions.csv` to `results/`.

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
- **Reproducibility:** `random_state=42` throughout.

## Results (Reported vs Ours)

**Classification, 2012 accuracy**

| Model | Reported | Ours |
|---|---|---|
| Logistic Regression | 0.8832 | 0.8832 |
| SVM | 0.8629 | 0.8680 |
| Gaussian Naive Bayes | 0.7614 | 0.8883 |
| MLP | 0.8782 | 0.8731 |
| Random Forest | 0.8731 | 0.8731 |

**Regression, 2012 RMSE**

| Model | Reported | Ours |
|---|---|---|
| Linear Regression | 3.1115 | 3.2068 |
| Ridge | 3.1582 | 3.1868 |
| Lasso | 3.0386 | 3.2515 |
| SVR | 2.8156 | 3.3439 |
| Poisson | 5.8460 | 5.5025 |
| Random Forest | 2.8869 | 2.5457 |
| Weighted Linear Regression | 6.6632 | 4.3064 |

**Final 2016 (Gaussian NB -> Ridge) RMSE:** reported 4.8243, ours 3.9663.

### Differences from the reference

Most models are within normal reproduction noise. Two differences are larger: Gaussian Naive Bayes (0.888 vs 0.761) and Weighted Linear Regression (4.31 vs 6.66). Likely reasons: the reference's exact preprocessing and weighting function are not fully specified, our merged dataset is rebuilt from public sources and may differ slightly from the paper's, and hyperparameter search ranges, random seeds and library versions differ. We report the final 2016 RMSE being lower than the reference as a reproduction difference, not as an improvement.

## Team contributions

| Member | Contribution |
|---|---|
| `Deepika K` | `Part 1 implementing classification and regression models according to the given paper` |
| `Disha Bose` | `Part 2. Implementing a new model and comaring with the previous chosen ones` |

## Library versions

Results were produced with the versions in `requirements.txt` (pandas, numpy, scikit-learn). Different scikit-learn versions can change results slightly.