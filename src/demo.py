import numpy as np
import pandas as pd
import streamlit as st
from config import RESULTS_DIR

st.set_page_config(page_title="Olympics Medal Prediction", layout="centered")
st.title("Olympics 2016: Predicted vs Actual Medals")

MODELS = {
    "Part 1: Gaussian NB + Ridge": RESULTS_DIR / "final_2016_predictions.csv",
    "Part 2: Gaussian NB + Gradient Boosting": RESULTS_DIR / "gradient_boosting_final_2016_predictions.csv",
}
available = {name: p for name, p in MODELS.items() if p.exists()}
if not available:
    st.error("No predictions found. Run `python src/02_run_reference_models.py` "
             "and `python src/gradient_boosting.py` first.")
    st.stop()

choice = st.selectbox("Model", list(available))
st.caption("Medal or not: Gaussian Naive Bayes. Medal count: the regressor named above. "
           "Trained on 1988-2012, tested on 2016.")

df = pd.read_csv(available[choice]).sort_values("Nation").reset_index(drop=True)
default = df.index[df.Nation == "USA"][0] if (df.Nation == "USA").any() else 0
country = st.selectbox("Pick a country (Olympic code)", df.Nation, index=int(default))
row = df[df.Nation == country].iloc[0]

actual, pred = int(row.Actual_Medals), float(row.Predicted_Medals)
c1, c2, c3 = st.columns(3)
c1.metric("Actual medals", actual)
c2.metric("Predicted medals", f"{pred:.1f}")
c3.metric("Error (pred - actual)", f"{pred - actual:+.1f}")

won, said_won = actual > 0, int(row.Predicted_Medal_Class) == 1
st.write(f"**Medal / no medal:** actual = {'medal' if won else 'no medal'}, "
         f"predicted = {'medal' if said_won else 'no medal'} "
         f"({'correct' if won == said_won else 'wrong'})")

st.bar_chart(pd.DataFrame({"Medals": [actual, pred]}, index=["Actual", "Predicted"]))

rmse = float(np.sqrt((df.Prediction_Error ** 2).mean()))
mae = float(df.Prediction_Error.abs().mean())
st.caption(f"Overall 2016 RMSE across {len(df)} countries: {rmse:.2f} | MAE: {mae:.2f}")

if len(available) == 2:
    rows = []
    for name, p in available.items():
        d = pd.read_csv(p)
        rows.append({"Model": name,
                     "RMSE": round(float(np.sqrt((d.Prediction_Error ** 2).mean())), 3),
                     "MAE": round(float(d.Prediction_Error.abs().mean()), 3)})
    with st.expander("Compare both models"):
        st.dataframe(pd.DataFrame(rows), hide_index=True)

with st.expander("Top 10 medal-winning countries in 2016"):
    st.dataframe(df.nlargest(10, "Actual_Medals")[["Nation", "Actual_Medals", "Predicted_Medals"]],
                 hide_index=True)