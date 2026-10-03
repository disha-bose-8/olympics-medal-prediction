"""Part 1 - reproduce the reference results (2012 validation table + 2016 final)."""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import (LogisticRegressionCV, LinearRegression, RidgeCV,
                                  LassoCV, PoissonRegressor)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from config import (DATA_PATH, RESULTS_DIR, TRAIN_YEARS, VALIDATION_YEAR,
                    TEST_YEAR, TARGET, FEATURES)

warnings.filterwarnings("ignore")
RS = 42
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

df = pd.read_csv(DATA_PATH).sort_values(["Year", "Nation"]).reset_index(drop=True)
tr, va, te = (df[df.Year.isin(TRAIN_YEARS)], df[df.Year == VALIDATION_YEAR],
              df[df.Year == TEST_YEAR])
Xt, yt = tr[FEATURES], tr[TARGET]
Xv, yv = va[FEATURES], va[TARGET]
Xe, ye = te[FEATURES], te[TARGET]
ct, cv_, ce = (yt > 0).astype(int), (yv > 0).astype(int), (ye > 0).astype(int)

# FIX 1: every model sits behind a StandardScaler (Pop ~1e9 vs GDP_Growth ~1)
S = StandardScaler

# ---------------- 1. CLASSIFICATION (2012 accuracy) ----------------
def make_classifiers():
    return {
        "Logistic Regression": make_pipeline(S(), LogisticRegressionCV(
            Cs=np.logspace(2, -2, 10), cv=3, scoring="accuracy",
            max_iter=10000, random_state=RS)),
        "SVM": make_pipeline(S(), GridSearchCV(      # FIX 2: tune C/gamma, no max_iter cap
            SVC(kernel="poly", degree=1),
            {"C": np.logspace(0, -2, 5), "gamma": np.logspace(0, -1, 5)},
            cv=3, scoring="accuracy", n_jobs=-1)),
        "Gaussian Naive Bayes": make_pipeline(S(), GridSearchCV(
            GaussianNB(), {"var_smoothing": np.logspace(0, -9, 10)},
            cv=3, scoring="accuracy", n_jobs=-1)),
        "MLP": make_pipeline(S(), RandomizedSearchCV(
            MLPClassifier(max_iter=1000, random_state=RS),
            {"activation": ["logistic", "relu", "tanh"],
             "hidden_layer_sizes": [(10,), (30,), (50,), (10, 30), (30, 50),
                                    (10, 30, 50), (100,), (100, 100)],
             "alpha": np.logspace(1, -4, 10),
             "learning_rate": ["constant", "invscaling", "adaptive"]},
            n_iter=10, cv=3, scoring="accuracy", random_state=RS, n_jobs=-1)),
        "Random Forest": RandomizedSearchCV(
            RandomForestClassifier(random_state=RS, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "criterion": ["gini", "entropy"],
             "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"],
             "bootstrap": [True, False]},
            n_iter=20, cv=3, scoring="accuracy", random_state=RS, n_jobs=-1),
    }

clf_rows = []
for name, m in make_classifiers().items():
    m.fit(Xt, ct)
    clf_rows.append([name, accuracy_score(cv_, m.predict(Xv))])
    print(f"{name}: {clf_rows[-1][1]:.4f}", flush=True)
clf_df = pd.DataFrame(clf_rows, columns=["Model", "Your_Accuracy"])

# ---------------- 2. REGRESSION (2012 RMSE) ----------------
# FIX 3: the reference table is plain regression on ALL training rows,
# raw (unrounded, unclipped) predictions, NO classifier gate.
al = np.linspace(0.1, 10.0, 15)
reg = {
    "Linear Regression": make_pipeline(S(), LinearRegression()),
    "Ridge": make_pipeline(S(), RidgeCV(alphas=al, cv=3)),
    "Lasso": make_pipeline(S(), LassoCV(alphas=np.linspace(0.01, 10, 30), cv=3, max_iter=10000)),
    "SVR": make_pipeline(S(), SVR(C=0.21544346900318834, kernel="linear", epsilon=0.1)),
    "Poisson": make_pipeline(S(), GridSearchCV(PoissonRegressor(max_iter=10000),
               {"alpha": al}, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)),
    # FIX 4: dropped criterion="absolute_error" (very slow, probably why the old run never finished)
    "Random Forest": RandomizedSearchCV(
        RandomForestRegressor(random_state=RS, n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
         "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
         "max_features": [1.0, "sqrt", "log2"], "bootstrap": [True, False]},
        n_iter=20, cv=3, scoring="neg_root_mean_squared_error", random_state=RS, n_jobs=-1),
}
reg_rows = []
for name, m in reg.items():
    m.fit(Xt, yt)
    reg_rows.append([name, rmse(yv, m.predict(Xv))])
    print(f"{name}: {reg_rows[-1][1]:.4f}", flush=True)
w = np.exp(-((yt - 50) ** 2) / 1000)
wl = make_pipeline(S(), LinearRegression()).fit(Xt, yt, linearregression__sample_weight=w)
reg_rows.append(["Weighted Linear Regression", rmse(yv, wl.predict(Xv))])
reg_df = pd.DataFrame(reg_rows, columns=["Model", "Your_RMSE"])

# ---------------- 3. FINAL 2016: Gaussian NB -> Ridge ----------------
# FIX 5: reference final model is fixed as GaussianNB gate + Ridge,
# trained on 1988-2012, tested on 2016 (not "best accuracy" auto-pick).
full = pd.concat([tr, va])
Xf, yf = full[FEATURES], full[TARGET]
gate = make_pipeline(S(), GridSearchCV(GaussianNB(), {"var_smoothing": np.logspace(0, -9, 10)},
                                       cv=3, scoring="accuracy")).fit(Xf, (yf > 0).astype(int))
ridge = make_pipeline(S(), RidgeCV(alphas=al, cv=3)).fit(Xf, yf)
cls = gate.predict(Xe)
pred = np.clip(np.where(cls == 1, ridge.predict(Xe), 0), 0, None)
final_rmse = rmse(ye, pred)
print(f"Final 2016 RMSE (GaussianNB -> Ridge): {final_rmse:.4f}")

out = te[["Nation"]].copy()
out["Actual_Medals"], out["Predicted_Medals"] = ye.values, pred
out["Predicted_Medal_Class"], out["Prediction_Error"] = cls, pred - ye.values
out.to_csv(RESULTS_DIR / "final_2016_predictions.csv", index=False)

# ---------------- 4. REPORTED vs YOURS tables ----------------
ref_c = pd.read_csv(RESULTS_DIR / "reference_classification_2012.csv").rename(columns={"Accuracy": "Reported_Accuracy"})
ref_c["Model"] = ref_c["Model"].replace({"Gaussian Naive Bayes": "Gaussian Naive Bayes"})
ref_r = pd.read_csv(RESULTS_DIR / "reference_regression_2012.csv").rename(columns={"RMSE": "Reported_RMSE"})
ref_r["Model"] = ref_r["Model"].replace({"Poisson": "Poisson"})
t1 = ref_c.merge(clf_df, on="Model"); t1["Diff"] = t1.Your_Accuracy - t1.Reported_Accuracy
t2 = ref_r.merge(reg_df, on="Model"); t2["Diff"] = t2.Your_RMSE - t2.Reported_RMSE
t3 = pd.DataFrame([["GaussianNB -> Ridge (2016)", pd.read_csv(RESULTS_DIR / "reference_final_2016_score.csv").RMSE[0], final_rmse]],
                  columns=["Model", "Reported_RMSE", "Your_RMSE"]); t3["Diff"] = t3.Your_RMSE - t3.Reported_RMSE
t1.to_csv(RESULTS_DIR / "compare_classification_2012.csv", index=False)
t2.to_csv(RESULTS_DIR / "compare_regression_2012.csv", index=False)
t3.to_csv(RESULTS_DIR / "compare_final_2016.csv", index=False)
print("\n", t1.round(4).to_string(index=False), "\n\n", t2.round(4).to_string(index=False),
      "\n\n", t3.round(4).to_string(index=False))