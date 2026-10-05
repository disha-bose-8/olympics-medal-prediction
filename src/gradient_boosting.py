import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV

from metrics import mae, rmse, selection_score, to_counts, top10_rmse

# ---------- load the shared dataset ----------
df = pd.read_csv("data/processed/olympics_merged.csv")

FEATURES = ["Year", "GDP", "GDP_Per_Capita", "GDP_Growth", "GDP_pct_World",
            "Pop", "Pop_pct_World", "Pop_Growth", "Area",
            "Athletes", "Athletes_pct", "Medals_Last_Games", "Total_Medals_Year"]
TARGET = "Medals"          # Nation is only a label, so it is not a feature

# ---------- the paper's split (Table 2) ----------
tune = df[df["Year"] <= 2008]       # 1988-2008: train and tune
val = df[df["Year"] == 2012]        # 2012: choose between models
test = df[df["Year"] == 2016]       # 2016: final score, used once at the end

X_tune, y_tune = tune[FEATURES], tune[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# 1 = won at least one medal, 0 = none (target for the classifier stage)
yb_tune = (y_tune > 0).astype(int)
yb_val = (y_val > 0).astype(int)
yb_test = (y_test > 0).astype(int)

# ---------- standardize features (fit on the tuning years only, then reuse) ----------
scaler = StandardScaler().fit(X_tune)
X_tune = pd.DataFrame(scaler.transform(X_tune), columns=FEATURES, index=X_tune.index)
X_val = pd.DataFrame(scaler.transform(X_val), columns=FEATURES, index=X_val.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)


def combine_stages(clf_pred, reg, X):
    """Classifier says 0 -> 0 medals. Classifier says 1 -> regressor's prediction,
    rounded to whole medals and clipped at 0 (as in the author's code)."""
    y_pred = np.zeros(len(X))
    mask = np.asarray(clf_pred) == 1
    if mask.any():
        y_pred[mask] = to_counts(reg.predict(X[mask]))
    return y_pred


def report(name, y_true, y_pred):
    print(f"{name:<26} RMSE {rmse(y_true, y_pred):.3f} | MAE {mae(y_true, y_pred):.3f} | "
          f"top-10 RMSE {top10_rmse(y_true, y_pred):.3f} | "
          f"selection score {selection_score(y_true, y_pred):.3f}")


if __name__ == "__main__":
    print("tune / val / test rows:", len(tune), len(val), len(test))

    # ----- stage 1: Gaussian Naive Bayes classifier (the paper's best classifier) -----
    gnb = GaussianNB().fit(X_tune, yb_tune)
    clf_val = gnb.predict(X_val)
    majority = max(yb_val.mean(), 1 - yb_val.mean())
    print("GNB accuracy on 2012:", round(accuracy_score(yb_val, clf_val), 3),
          "| paper: 0.891 | always guessing the majority class:", round(majority, 3))

    # ----- stage 2 stand-in: Ridge trained on rows that really won medals -----
    pos = y_tune > 0
    ridge = RidgeCV(alphas=np.linspace(0.1, 10.0, 30)).fit(X_tune[pos], y_tune[pos])

    y_val_pred = combine_stages(clf_val, ridge, X_val)
    report("GNB + Ridge, 2012 val", y_val, y_val_pred)
    print("paper reports 2.27 for Ridge on the 2012 validation set (RMSE)")
    # ===== Step 4: Gradient Boosting, default settings =====
    print("\n--- Gradient Boosting (default settings) ---")

    gb_clf = GradientBoostingClassifier(random_state=42).fit(X_tune, yb_tune)
    gb_clf_val = gb_clf.predict(X_val)
    print("GB classifier accuracy on 2012:", round(accuracy_score(yb_val, gb_clf_val), 3),
          "| GNB:", round(accuracy_score(yb_val, clf_val), 3))

    gb_reg = GradientBoostingRegressor(random_state=42).fit(X_tune[pos], y_tune[pos])

    report("GNB + Ridge", y_val, y_val_pred)
    report("GNB + GradientBoosting", y_val, combine_stages(clf_val, gb_reg, X_val))
    report("GB clf + GB reg", y_val, combine_stages(gb_clf_val, gb_reg, X_val))

    # ===== Step 5: tune Gradient Boosting (random search, 3-fold CV on 1988-2008 only) =====
    print("\n--- Tuning Gradient Boosting ---")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)   # shuffled, as the paper describes
    param_dist = {
        "n_estimators": randint(100, 600),
        "learning_rate": uniform(0.01, 0.19),     # 0.01 to 0.20
        "max_depth": randint(2, 6),               # 2 to 5
        "min_samples_leaf": randint(1, 8),
        "subsample": uniform(0.6, 0.4),           # 0.6 to 1.0
        "max_features": ["sqrt", None],
    }

    rs_clf = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_dist,
                                n_iter=40, cv=cv, scoring="accuracy",
                                random_state=42, n_jobs=-1).fit(X_tune, yb_tune)
    rs_reg = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_dist,
                                n_iter=40, cv=cv, scoring="neg_mean_squared_error",
                                random_state=42, n_jobs=-1).fit(X_tune[pos], y_tune[pos])

    tuned_clf, tuned_reg = rs_clf.best_estimator_, rs_reg.best_estimator_
    print("best classifier params:", rs_clf.best_params_)
    print("best regressor params:", rs_reg.best_params_)

    tuned_clf_val = tuned_clf.predict(X_val)
    print("tuned GB classifier accuracy on 2012:", round(accuracy_score(yb_val, tuned_clf_val), 3))

    report("GNB + Ridge", y_val, y_val_pred)
    report("GNB + tuned GB reg", y_val, combine_stages(clf_val, tuned_reg, X_val))
    report("tuned GB clf + reg", y_val, combine_stages(tuned_clf_val, tuned_reg, X_val))

    # diagnosis: what happens to the biggest medal winners?
    show = pd.DataFrame({"Nation": val["Nation"].values, "Actual": y_val.values,
                         "Ridge": y_val_pred,
                         "GB": combine_stages(clf_val, tuned_reg, X_val)})
    print(show.sort_values("Actual", ascending=False).head(10).to_string(index=False))