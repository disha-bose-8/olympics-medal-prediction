"""Part 2: Gradient Boosting, our extra model, compared with the Part 1 models.

Run from the repo root:  python src/gradient_boosting.py
"""
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from config import (DATA_PATH, FEATURES, RESULTS_DIR, TARGET,
                    TEST_YEAR, TRAIN_YEARS, VALIDATION_YEAR)
from metrics import mae, rmse, selection_score, top10_rmse

# ---------- 1. load the data and split it by year (same split as Part 1) ----------
df = pd.read_csv(DATA_PATH).sort_values(["Year", "Nation"]).reset_index(drop=True)
train = df[df["Year"].isin(TRAIN_YEARS)]    # 1988-2008: train and tune here
val = df[df["Year"] == VALIDATION_YEAR]     # 2012: compare models here
test = df[df["Year"] == TEST_YEAR]          # 2016: final test, used once at the end


def scale(fit_on, other):
    """Put every feature on a similar scale. The scaler only learns from fit_on."""
    scaler = StandardScaler().fit(fit_on[FEATURES])
    return scaler.transform(fit_on[FEATURES]), scaler.transform(other[FEATURES])


def make_gate():
    """Step 1 of the model: will this country win any medal? (same Gaussian NB as Part 1)"""
    return GridSearchCV(GaussianNB(), {"var_smoothing": np.logspace(0, -9, 10)},
                        cv=3, scoring="accuracy")


def two_stage(gate_says, regressor, X, last_games=None):
    """Gate says 'no medal' -> 0. Otherwise use the regressor's number.
    last_games is only used by the change-target version, which predicts a change."""
    pred = np.zeros(len(X))
    medal = gate_says == 1
    guess = regressor.predict(X[medal])
    if last_games is not None:
        guess = guess + last_games[medal]
    pred[medal] = np.clip(guess, 0, None)   # no negative medals
    return pred


def show(name, actual, predicted):
    """Print the four scores for one model and return them as a row for the CSV."""
    row = {"Model": name, "RMSE": rmse(actual, predicted), "MAE": mae(actual, predicted),
           "Top10_RMSE": top10_rmse(actual, predicted),
           "Selection_score": selection_score(actual, predicted)}
    print(f"{name:<32} RMSE {row['RMSE']:.3f} | MAE {row['MAE']:.3f} | "
          f"top-10 RMSE {row['Top10_RMSE']:.3f} | score {row['Selection_score']:.3f}")
    return row


# ---------- 2. prepare the training and validation data ----------
X_train, X_val = scale(train, val)
y_train, y_val = train[TARGET].values, val[TARGET].values
won_train = (y_train > 0).astype(int)       # 1 = won a medal, 0 = won none
won_val = (y_val > 0).astype(int)
has_medals = y_train > 0                    # rows of countries that won something
last_train = train["Medals_Last_Games"].values
last_val = val["Medals_Last_Games"].values

# ---------- 3. classification on 2012: gate models ----------
gnb = make_gate().fit(X_train, won_train)
gate_val = gnb.predict(X_val)

# one random search setup, reused for every Gradient Boosting model below
search_space = {"n_estimators": randint(100, 600), "learning_rate": uniform(0.01, 0.19),
                "max_depth": randint(2, 6), "min_samples_leaf": randint(1, 8),
                "subsample": uniform(0.6, 0.4), "max_features": ["sqrt", None]}
folds = KFold(n_splits=3, shuffle=True, random_state=42)

clf_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), search_space,
                                n_iter=40, cv=folds, scoring="accuracy", random_state=42,
                                n_jobs=-1).fit(X_train, won_train)
acc_gnb = accuracy_score(won_val, gate_val)
acc_gb = accuracy_score(won_val, clf_search.predict(X_val))
print(f"2012 accuracy: Gaussian NB {acc_gnb:.4f} | Gradient Boosting {acc_gb:.4f}")
pd.DataFrame({"Model": ["Gaussian NB", "Gradient Boosting"],
              "Accuracy_2012": [acc_gnb, acc_gb]}).to_csv(
    RESULTS_DIR / "gradient_boosting_classification_2012.csv", index=False)

# ---------- 4. regression on 2012 ----------
# (a) plain: Gradient Boosting on all training rows, no gate
plain_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), search_space,
                                  n_iter=40, cv=folds, scoring="neg_root_mean_squared_error",
                                  random_state=42, n_jobs=-1).fit(X_train, y_train)
plain_rmse = rmse(y_val, plain_search.predict(X_val))
print(f"Plain Gradient Boosting regression, 2012 RMSE: {plain_rmse:.4f}")
pd.DataFrame({"Model": ["Gradient Boosting"], "Your_RMSE": [plain_rmse]}).to_csv(
    RESULTS_DIR / "gradient_boosting_plain_regression_2012.csv", index=False)

# (b) two-stage: gate first, then a regressor trained only on medal-winning countries
reg_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), search_space,
                                n_iter=40, cv=folds, scoring="neg_mean_squared_error",
                                random_state=42, n_jobs=-1).fit(X_train[has_medals],
                                                                y_train[has_medals])
best = reg_search.best_params_
print("best regressor settings:", best)

ridge = RidgeCV(alphas=np.linspace(0.1, 10, 30)).fit(X_train[has_medals], y_train[has_medals])
# change-target version: learn "medals now minus medals last Games", add last Games back later
change = GradientBoostingRegressor(random_state=42, **best).fit(
    X_train[has_medals], (y_train - last_train)[has_medals])

print("\nTwo-stage models on 2012 (Gaussian NB gate):")
rows = [show("GNB + Ridge", y_val, two_stage(gate_val, ridge, X_val)),
        show("GNB + GB", y_val, two_stage(gate_val, reg_search.best_estimator_, X_val)),
        show("GNB + GB (change target)", y_val, two_stage(gate_val, change, X_val, last_val))]
pd.DataFrame(rows).to_csv(RESULTS_DIR / "gradient_boosting_compare_regression_2012.csv", index=False)

# pick the better Gradient Boosting version using 2012 only (lower score is better)
use_change = rows[2]["Selection_score"] < rows[1]["Selection_score"]
print("Version chosen on 2012:", "change target" if use_change else "plain")

# ---------- 5. final test on 2016: train on 1988-2012, test once ----------
train_val = pd.concat([train, val])
X_tv, X_test = scale(train_val, test)
y_tv, y_test = train_val[TARGET].values, test[TARGET].values
last_tv, last_test = train_val["Medals_Last_Games"].values, test["Medals_Last_Games"].values
medals_tv = y_tv > 0

gate_test = make_gate().fit(X_tv, medals_tv.astype(int)).predict(X_test)
ridge_final = RidgeCV(alphas=np.linspace(0.1, 10, 30)).fit(X_tv[medals_tv], y_tv[medals_tv])

target = (y_tv - last_tv) if use_change else y_tv
gb_final = GradientBoostingRegressor(random_state=42, **best).fit(X_tv[medals_tv],
                                                                  target[medals_tv])
pred_gb = two_stage(gate_test, gb_final, X_test, last_test if use_change else None)
pred_ridge = two_stage(gate_test, ridge_final, X_test)

# Part 1's saved 2016 predictions (its Ridge uses all rows, not only medal winners)
part1 = pd.read_csv(RESULTS_DIR / "final_2016_predictions.csv").set_index("Nation")
part1_pred = part1.loc[test["Nation"], "Predicted_Medals"].values

print("\nFinal 2016 test:")
rows = [show("Part 1: GNB + Ridge", y_test, part1_pred),
        show("GNB + Ridge (medal rows only)", y_test, pred_ridge),
        show("Part 2: GNB + Gradient Boosting", y_test, pred_gb)]
pd.DataFrame(rows).to_csv(RESULTS_DIR / "gradient_boosting_final_2016_score.csv", index=False)

out = pd.DataFrame({"Nation": test["Nation"].values, "Actual_Medals": y_test,
                    "Predicted_Medals": pred_gb, "Predicted_Medal_Class": gate_test,
                    "Prediction_Error": pred_gb - y_test})
out.to_csv(RESULTS_DIR / "gradient_boosting_final_2016_predictions.csv", index=False)
print(out.sort_values("Actual_Medals", ascending=False).head(10).to_string(index=False))