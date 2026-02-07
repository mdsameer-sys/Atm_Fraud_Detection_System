import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="ATM Fraud Detection",
    page_icon="🏧",
    layout="wide"
)

# --------------------------------------------------
# Expected schema (VERY IMPORTANT)
# --------------------------------------------------
EXPECTED_COLUMNS = [
    'Per1','Per2','Per3','Per4','Per5','Per6','Per7','Per8','Per9',
    'Dem1','Dem2','Dem3','Dem4','Dem5','Dem6','Dem7','Dem8','Dem9',
    'Cred1','Cred2','Cred3','Cred4','Cred5','Cred6',
    'Normalised_FNT',
    'geo_score_mean','geo_score_max',
    'instance_score_mean','instance_score_max',
    'lambda_wt',
    'qsets_normalized_tat_mean','qsets_normalized_tat_max'
]
# --------------------------------------------------
# Load model & threshold
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("stacking_fraud_pipeline.pkl")
    threshold = joblib.load("best_stack_threshold.pkl")
    return pipeline, threshold

model, default_threshold = load_artifacts()

st.sidebar.header("🎛 Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01
)

st.sidebar.markdown(
    f"Current threshold: **{threshold:.2f}**"
)

st.sidebar.warning(
    "⚠ Changing threshold affects precision & recall trade-off."
)

st.sidebar.markdown("---")
st.sidebar.success("✅ Model loaded successfully")



# --------------------------------------------------
# Main header
# --------------------------------------------------
st.title("🏧 ATM Fraud Detection – Stacking Model")

st.success("✔ Model and preprocessing pipeline loaded")

# --------------------------------------------------
# File upload
# --------------------------------------------------
st.markdown("### 📂 Upload Transaction Data (CSV)")

uploaded_file = st.file_uploader(
    "CSV must contain the same columns used during training",
    type=["csv"]
)

# --------------------------------------------------
# Helper: schema validation
# --------------------------------------------------
def validate_and_prepare_input(df):
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Drop extra columns safely
    if extra:
        df = df.drop(columns=list(extra))

    # Reorder columns exactly as training
    df = df[EXPECTED_COLUMNS]

    return df

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(raw_df.head())

        # ▶ Run button
        run_button = st.button("▶ Run Predictions")

        if run_button:

            # Validate & align schema
            input_df = validate_and_prepare_input(raw_df)

            # Predict
            probs = model.predict_proba(input_df)[:, 1]
            preds = (probs >= threshold).astype(int)

        
            final_output_df = pd.DataFrame({
                "Fraud_Prediction": preds.astype(int)
            })

            # Results
            result_df = raw_df.copy()
            result_df["Fraud_Probability"] = probs

            # ✅ BINARY PREDICTION (0 / 1)
            result_df["Fraud_Prediction"] = preds.astype(int)

            total_txns = len(result_df)
            fraud_count = int((preds == 1).sum())
            fraud_pct = (fraud_count / total_txns) * 100
            
            display_cols = (
                [c for c in result_df.columns
                if c not in ["Fraud_Prediction", "Fraud_Probability"]]
                + ["Fraud_Prediction", "Fraud_Probability"]
            )
            st.markdown("### 🔍 Prediction Results")
            st.dataframe(result_df[display_cols].head(20))


            # Summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", f"{total_txns}")
            col2.metric(
                "Fraudulent Detected",
                f"{fraud_count}",
                help="Number of transactions classified as fraud"
            )
            col3.metric(
                "Fraud Percentage",
                f"{fraud_pct:.2f}%",
                help="Fraud transactions as % of total"
            )

            # Download
            st.markdown("### ⬇ Download Predictions")
            st.download_button(
                label="Download CSV (0 = Legit, 1 = Fraud)",
                data=final_output_df.to_csv(index=False),
                file_name="atm_fraud_Final_values.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Model: Stacking Classifier | ATM Fraud Detection System"
)
