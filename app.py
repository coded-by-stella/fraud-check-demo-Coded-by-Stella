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

**What it does**  
Upload a CSV of transactions and instantly see which ones are *Suspicious* vs *Legit*.  
No waiting, no manual checks, just clear results. A chart that tells the story and a file you can download in one click.  

**Stack**  
Python · pandas · scikit-learn · Streamlit · matplotlib  

**Why it matters** 
             
Every business handling digital payments or customer data faces the same threats:  
❌ Fraud that drains revenue  
❌ Endless manual reviews that slow down teams  
❌ Blind spots that lead to compliance risks  

This demo shows how I can turn raw data into a tool that solves those pain points:  

⚡ **Faster decisions** → predictions in seconds, not hours  
📊 **Clarity for everyone** → insights that even non-technical staff can act on  
💾 **Proof on demand** → exportable reports ready for audits or management  

👉 Imagine giving your risk team a dashboard that flags threats *before* they turn into losses.  
That’s the power of building AI-driven apps like this: simple to use, but critical for protecting your business.
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

