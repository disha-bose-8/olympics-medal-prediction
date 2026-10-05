"""Part 2: Gradient Boosting as the additional model.

Same split (Table 2 of the paper), same features (src/config.py) and same metrics
(src/metrics.py) as Part 1, so the results can sit in the same comparison tables.

Run from the repo root:  python src/gradient_boosting.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from config import (DATA_PATH, FEATURES as BASE_FEATURES, RESULTS_DIR, TARGET,
                    TEST_YEAR, TRAIN_YEARS, VALIDATION_YEAR)
from metrics import mae, rmse, selection_score, top10_rmse

warnings.filterwarnings("ignore")   # sklearn FutureWarnings from LogisticRegressionCV

# Features come from src/config.py (now the paper's 13). If config.py still lists only 9,
# set USE_PAPER_13 = True to add the other four here without editing config.py.
USE_PAPER_13 = False
FEATURES = list(BASE_FEATURES)
if USE_PAPER_13:
    FEATURES += ["GDP", "GDP_pct_World", "Pop_Growth", "Area"]
FEATURES = list(dict.fromkeys(FEATURES))   # drop duplicates, keep order

# Values printed in the paper (Dobkowski), for reference only
PAPER_ACC = {"Logistic Reg.": 0.877, "SVM Clf": 0.884, "Random Forest": 0.877,
             "Gaussian NB": 0.891, "MLP": 0.855}
PAPER_RMSE = {"Ridge": 2.27, "Random Forest": 2.20, "SVM Reg": 2.50, "Final 2016": 2.26}

# ---------- data and the paper's split ----------
# same row order as Part 1 (its un-shuffled 3-fold CV depends on it)
df = pd.read_csv(DATA_PATH).sort_values(["Year", "Nation"]).reset_index(drop=True)
tune = df[df["Year"].isin(TRAIN_YEARS)]
val = df[df["Year"] == VALIDATION_YEAR]
test = df[df["Year"] == TEST_YEAR]


def scale(train, *others):
    """Fit the scaler on `train` only, return scaled DataFrames for train and others."""
    sc = StandardScaler().fit(train[FEATURES])
    return [pd.DataFrame(sc.transform(d[FEATURES]), columns=FEATURES, index=d.index)
            for d in (train, *others)]


def combine_stages(clf_pred, reg, X, offset=None):
    """Classifier says 0 -> 0 medals. Says 1 -> regressor prediction (plus offset if the
    regressor predicts a change), clipped at 0. Not rounded, same as Part 1."""
    y_pred = np.zeros(len(X))
    mask = np.asarray(clf_pred) == 1
    if mask.any():
        raw = reg.predict(X[mask])
        if offset is not None:
            raw = raw + np.asarray(offset)[mask]
        y_pred[mask] = np.clip(raw, 0, None)
    return y_pred


def report(name, y_true, y_pred):
    row = {"Model": name, "RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
           "Top10_RMSE": top10_rmse(y_true, y_pred),
           "Selection_score": selection_score(y_true, y_pred)}
    print(f"{name:<30} RMSE {row['RMSE']:.3f} | MAE {row['MAE']:.3f} | "
          f"top-10 RMSE {row['Top10_RMSE']:.3f} | selection score {row['Selection_score']:.3f}")
    return row


def part1_gnb():
    """Same Gaussian NB as Part 1: var_smoothing tuned with 3-fold CV."""
    return GridSearchCV(GaussianNB(), {"var_smoothing": np.logspace(0, -9, 10)},
                        cv=3, scoring="accuracy", n_jobs=-1)


def part1_lr():
    """Same Logistic Regression as Part 1."""
    return LogisticRegressionCV(Cs=np.logspace(2, -2, 10), cv=3, scoring="accuracy",
                                max_iter=10000, random_state=42)


def clean(params):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, np.integer) else v) for k, v in params.items()}


if __name__ == "__main__":
    print("features used:", len(FEATURES), "| tune / val / test rows:", len(tune), len(val), len(test))
    X_tune, X_val = scale(tune, val)
    y_tune, y_val = tune[TARGET], val[TARGET]
    yb_tune, yb_val = (y_tune > 0).astype(int), (y_val > 0).astype(int)
    pos = (y_tune > 0).values
    last_val = val["Medals_Last_Games"].values
    last_tune = tune["Medals_Last_Games"].values

    # ===== 1. classification on 2012 =====
    print("\n--- Classification, 2012 accuracy ---")
    gnb = part1_gnb().fit(X_tune, yb_tune)
    lr = part1_lr().fit(X_tune, yb_tune)
    gb_clf = GradientBoostingClassifier(random_state=42).fit(X_tune, yb_tune)
    clf_preds = {"GNB": gnb.predict(X_val), "LR": lr.predict(X_val), "GB": gb_clf.predict(X_val)}
    print("majority-class baseline:", round(max(yb_val.mean(), 1 - yb_val.mean()), 3))
    for k, p in clf_preds.items():
        print(f"{k} accuracy: {accuracy_score(yb_val, p):.4f}")
    print("check: these GNB and LR numbers must equal the ones printed by 02_run_reference_models.py")
    print("paper Table 3: GNB 0.891, LR 0.877")
    acc_rows = [{"Model": k, "Accuracy_2012": accuracy_score(yb_val, p)} for k, p in clf_preds.items()]

    # ===== 2. tune Gradient Boosting (random search, 3-fold shuffled CV, 1988-2008 only) =====
    print("\n--- Tuning Gradient Boosting ---")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    param_dist = {"n_estimators": randint(100, 600), "learning_rate": uniform(0.01, 0.19),
                  "max_depth": randint(2, 6), "min_samples_leaf": randint(1, 8),
                  "subsample": uniform(0.6, 0.4), "max_features": ["sqrt", None]}
    rs_clf = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_dist, n_iter=40,
                                cv=cv, scoring="accuracy", random_state=42, n_jobs=-1).fit(X_tune, yb_tune)
    rs_reg = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_dist, n_iter=40,
                                cv=cv, scoring="neg_mean_squared_error", random_state=42,
                                n_jobs=-1).fit(X_tune[pos], y_tune[pos])
    tuned_clf, tuned_reg = rs_clf.best_estimator_, rs_reg.best_estimator_
    print("best classifier params:", clean(rs_clf.best_params_))
    print("best regressor params:", clean(rs_reg.best_params_))
    clf_preds["GB tuned"] = tuned_clf.predict(X_val)
    print(f"tuned GB classifier accuracy 2012: {accuracy_score(yb_val, clf_preds['GB tuned']):.4f}")
    acc_rows.append({"Model": "GB tuned", "Accuracy_2012": accuracy_score(yb_val, clf_preds["GB tuned"])})

    # ===== 2b. plain GB regressor on ALL training rows, raw predictions (Part 1's table format) =====
    rs_plain = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_dist, n_iter=40,
                                  cv=cv, scoring="neg_root_mean_squared_error", random_state=42,
                                  n_jobs=-1).fit(X_tune, y_tune)
    plain_rmse = rmse(y_val, rs_plain.best_estimator_.predict(X_val))
    print(f"\nPlain Gradient Boosting regressor, 2012 RMSE: {plain_rmse:.4f} "
          "(compare with the 2012 table in results/compare_regression_2012.csv)")
    pd.DataFrame([{"Model": "Gradient Boosting", "Your_RMSE": plain_rmse}]).to_csv(
        RESULTS_DIR / "gradient_boosting_plain_regression_2012.csv", index=False)

    # ===== 3. regression on 2012, two-stage (classifier gate + regressor) =====
    # Paper section 5.3: regressors in Table 3 are fitted on rows the classifier says are 1.
    print("\n--- Regression 2012: gate + regressor ---")
    ridge = RidgeCV(alphas=np.linspace(0.1, 10.0, 30)).fit(X_tune[pos], y_tune[pos])
    gb_default = GradientBoostingRegressor(random_state=42).fit(X_tune[pos], y_tune[pos])
    # experiment: predict the CHANGE from last games' medals, then add it back
    delta = clone(tuned_reg).fit(X_tune[pos], (y_tune.values - last_tune)[pos])
    regs = [("Ridge", ridge, None), ("GB default", gb_default, None),
            ("GB tuned", tuned_reg, None), ("GB tuned (change target)", delta, last_val)]
    rows, val_pred = [], {}
    for gate in ["LR", "GNB"]:
        for name, reg, off in regs:
            pred = combine_stages(clf_preds[gate], reg, X_val, off)
            val_pred[(gate, name)] = pred
            rows.append(report(f"{gate} + {name}", y_val, pred))
    print("paper (LR gate): Ridge 2.27, Random Forest 2.20, SVM Reg 2.50")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "gradient_boosting_compare_regression_2012.csv", index=False)
    pd.DataFrame(acc_rows).to_csv(RESULTS_DIR / "gradient_boosting_classification_2012.csv", index=False)

    # ===== 4. final 2016 test (GNB gate, as in the paper), trained on 1988-2012 =====
    # Choose plain vs change-target GB on 2012 only, then touch 2016 once.
    use_delta = (selection_score(y_val, val_pred[("GNB", "GB tuned (change target)")])
                 < selection_score(y_val, val_pred[("GNB", "GB tuned")]))
    print("\n--- Final 2016 test (trained on 1988-2012) ---")
    print("GB variant chosen on 2012:", "change target" if use_delta else "raw medals")
    trainval = pd.concat([tune, val])
    Xtv, Xte = scale(trainval, test)
    ytv, yte = trainval[TARGET], test[TARGET]
    pos_tv = (ytv > 0).values
    last_tv, last_te = trainval["Medals_Last_Games"].values, test["Medals_Last_Games"].values

    clf_te = part1_gnb().fit(Xtv, (ytv > 0).astype(int)).predict(Xte)
    ridge_f = RidgeCV(alphas=np.linspace(0.1, 10.0, 30)).fit(Xtv[pos_tv], ytv[pos_tv])
    tgt = (ytv.values - last_tv) if use_delta else ytv.values
    gb_f = clone(tuned_reg).fit(Xtv[pos_tv], tgt[pos_tv])

    pred_ridge = combine_stages(clf_te, ridge_f, Xte)
    pred_gb = combine_stages(clf_te, gb_f, Xte, last_te if use_delta else None)
    # Part 1's own saved predictions (its Ridge is trained on ALL rows, not only medal winners)
    p1 = pd.read_csv(RESULTS_DIR / "final_2016_predictions.csv").set_index("Nation")
    p1_pred = p1.loc[test["Nation"], "Predicted_Medals"].values
    score_rows = [report("Part 1: GNB + Ridge (2016)", yte, p1_pred),
                  report("GNB + Ridge, medal rows only", yte, pred_ridge),
                  report("GNB + Gradient Boosting (2016)", yte, pred_gb)]
    print("paper final: 2.26 (note: its Table 4 'actual' values look like 2012 data)")
    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "gradient_boosting_final_2016_score.csv", index=False)
    out = pd.DataFrame({"Nation": test["Nation"].values, "Actual_Medals": yte.values,
                        "Predicted_Medals": pred_gb, "Predicted_Medal_Class": clf_te,
                        "Prediction_Error": pred_gb - yte.values})
    out.to_csv(RESULTS_DIR / "gradient_boosting_final_2016_predictions.csv", index=False)
    print(out.sort_values("Actual_Medals", ascending=False).head(10).to_string(index=False))
    print("\nsaved CSVs to", RESULTS_DIR)