
import numpy as np
import pandas as pd
import streamlit as st
from config import RESULTS_DIR

st.set_page_config(page_title="Olympics Medal Prediction", layout="centered")
st.title("Olympics 2016: Predicted vs Actual Medals")
st.caption("Model: Gaussian Naive Bayes (medal or not) + Ridge regression (medal count), trained on 1988-2012.")

path = RESULTS_DIR / "final_2016_predictions.csv"
if not path.exists():
    st.error("Run `python src/02_run_reference_models.py` first to create the predictions file.")
    st.stop()

df = pd.read_csv(path).sort_values("Nation").reset_index(drop=True)
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
st.caption(f"Overall 2016 RMSE across {len(df)} countries: {rmse:.2f}")
with st.expander("Top 10 medal-winning countries in 2016"):
    st.dataframe(df.nlargest(10, "Actual_Medals")[["Nation", "Actual_Medals", "Predicted_Medals"]],
                 hide_index=True)