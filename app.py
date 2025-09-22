# app.py Streamlit web UI
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Ensure data/model exist on first run (Cloud-safe) 
import sys, subprocess
from pathlib import Path

if not Path("transactions.csv").exists():
    try:
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    except Exception as e:
        st.warning(f"Could not generate transactions.csv automatically: {e}")

if not Path("model.joblib").exists():
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
    except Exception as e:
        st.warning(f"Could not train model automatically: {e}")
# ---------------------------------------------------------

st.set_page_config(page_title="Fraud Check Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Check Dashboard (Demo)")

st.markdown("""
### 👩‍💻 Project Overview

**What it does:** Upload a CSV of transactions, run a simple ML model to flag *Suspicious* vs *Legit*, view predictions, see a quick chart and download results.  

**Stack:** Python · pandas · scikit-learn · Streamlit · matplotlib  

**Why it matters:**  
Fraud and risk detection tools help companies **save time, reduce financial losses and support compliance**.  
Even with a lightweight demo, this project shows how I can transform raw data into a usable tool that:  
- ⚡ Speeds up decision-making with instant predictions  
- 📊 Provides clear visual insights for non-technical staff  
- 💾 Exports results for reporting or audit purposes  
""")
st.divider()

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_default_csv():
    return pd.read_csv("transactions.csv")

model = load_model()

# - Sidebar / Data 
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("Using bundled sample: transactions.csv")
    df = load_default_csv()

st.subheader("Input data")
st.write(df.head())

# - Sidebar / Predict controls 
st.sidebar.header("2) Predict")
threshold = st.sidebar.slider("Decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

if st.sidebar.button("Run prediction"):
    # Predict
    X = df[["amount", "country", "channel", "merchant_id", "hour_of_day", "is_new_device", "ip_risk_score"]]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_score"] = (proba * 100).round(1)
    out["prediction"] = np.where(pred == 1, "Suspicious", "Legit")

    # Summary 
    total = len(out)
    suspicious = int((out["prediction"] == "Suspicious").sum())
    pct = (suspicious / total * 100) if total else 0
    st.caption(f"Summary: {suspicious} of {total} transactions flagged as suspicious ({pct:.1f}%) at threshold {threshold:.2f}.")

    # Table with highlight
    st.subheader("Predictions")
    def _row_style(row):
        color = "#ffe5e5" if row.get("prediction") == "Suspicious" else ""
        return [f"background-color: {color}"] * len(row)

    styled = out.head(50).style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Chart
    st.subheader("Overview")
    counts = out["prediction"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)  
    ax.set_ylabel("Count")
    ax.set_title("Predicted classes")
    st.pyplot(fig)

    # Export
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Click **Run prediction** in the sidebar to score your data.")

