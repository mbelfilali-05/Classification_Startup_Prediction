"""
app/streamlit_app.py
─────────────────────────────────────────────────────────────────────
Streamlit Demo UI for Startup Classification.

This app uses the EXACT SAME pipeline as training:
  RawCleaner → FeatureEngineer → CategoryGrouper → FeatureSelector → Preprocessor → Model

No feature engineering is duplicated — everything flows through the
saved transformers from nb02.

Usage:
  cd startup-classification
  streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from utils.feature_engineering import (
    RawCleaner, FeatureEngineer, CategoryGrouper, FeatureSelector,
    ALL_FEATURE_COLS,
)


# ═══════════════════════════════════════════════════════════════════
# Load artifacts (cached so they load only once)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """Load all saved pipeline components and model."""
    base = os.path.join(os.path.dirname(__file__), '..')
    artifacts = os.path.join(base, 'artifacts')
    models = os.path.join(base, 'models')

    return {
        'cleaner': joblib.load(os.path.join(artifacts, 'cleaner.pkl')),
        'engineer': joblib.load(os.path.join(artifacts, 'engineer.pkl')),
        'grouper': joblib.load(os.path.join(artifacts, 'grouper.pkl')),
        'selector': joblib.load(os.path.join(artifacts, 'selector.pkl')),
        'preprocessor': joblib.load(os.path.join(artifacts, 'preprocessor.pkl')),
        'label_encoder': joblib.load(os.path.join(artifacts, 'label_encoder_target.pkl')),
        'model': joblib.load(os.path.join(models, 'xgb_model.pkl')),
        'top_countries': joblib.load(os.path.join(artifacts, 'top_countries.pkl')),
        'top_categories': joblib.load(os.path.join(artifacts, 'top_categories.pkl')),
        'top_regions': joblib.load(os.path.join(artifacts, 'top_regions.pkl')),
    }


def predict_single(row_dict, art):
    """
    Run the full pipeline on a single startup's data.

    This is the EXACT same pipeline as training:
    RawCleaner → FeatureEngineer → CategoryGrouper → FeatureSelector → Preprocessor → Model
    """
    df = pd.DataFrame([row_dict])

    df = art['cleaner'].transform(df)
    df = art['engineer'].transform(df)
    df = art['grouper'].transform(df)
    df = art['selector'].transform(df)

    X = art['preprocessor'].transform(df)
    proba = art['model'].predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_label = art['label_encoder'].classes_[pred_idx]

    return pred_label, proba, art['label_encoder'].classes_


def predict_batch(uploaded_df, art):
    """Run the full pipeline on a batch of startups from CSV upload."""
    df = art['cleaner'].transform(uploaded_df)
    df = art['engineer'].transform(df)
    df = art['grouper'].transform(df)
    df = art['selector'].transform(df)

    X = art['preprocessor'].transform(df)
    proba = art['model'].predict_proba(X)
    preds = art['model'].predict(X)
    labels = art['label_encoder'].inverse_transform(preds)

    results = uploaded_df[['name']].copy() if 'name' in uploaded_df.columns else pd.DataFrame()
    results['predicted_status'] = labels
    for i, cls in enumerate(art['label_encoder'].classes_):
        results[f'prob_{cls}'] = proba[:, i].round(4)

    return results


# ═══════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Startup Classifier",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 Startup Classification: DL vs ML")
    st.markdown(
        "Predict startup outcomes (**operating**, **acquired**, **closed**, **ipo**) "
        "using the best model from our ML vs DL comparison."
    )
    st.markdown("---")

    # Load artifacts
    try:
        art = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model artifacts. Run nb02 and nb03 first.\n\nError: {e}")
        return

    # ── Tabs ──
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction (CSV)"])

    # ────────────────────────────────────────────
    # TAB 1: Single Prediction
    # ────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Funding")
            funding = st.number_input("Total Funding (USD)", min_value=0, value=5_000_000, step=100_000)
            rounds = st.number_input("Funding Rounds", min_value=0, max_value=20, value=2)

        with col2:
            st.subheader("Company Info")
            categories = ["Software", "Biotechnology", "Mobile", "E-Commerce",
                          "Health Care", "Enterprise Software", "Education",
                          "Analytics", "SaaS", "Other"]
            category = st.selectbox("Primary Category", categories)
            country_options = art['top_countries'] + ["Other"]
            country = st.selectbox("Country", country_options, index=0)
            region_options = art['top_regions'] + ["Other"]
            region = st.selectbox("Region", region_options, index=0)

        with col3:
            st.subheader("Dates & Other")
            founded_year = st.number_input("Founded Year", min_value=1990, max_value=2025, value=2018)
            has_website = st.checkbox("Has Website?", value=True)
            multi_cat = st.checkbox("Multiple Categories?", value=False)
            category_list_str = f"{category}|Other" if multi_cat else category

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            # Build a raw row matching the original CSV columns
            row = {
                'name': 'User Input',
                'homepage_url': 'http://example.com' if has_website else np.nan,
                'category_list': category_list_str,
                'funding_total_usd': str(funding),
                'status': 'unknown',
                'country_code': country,
                'state_code': '',
                'region': region,
                'city': '',
                'funding_rounds': rounds,
                'founded_at': f'{founded_year}-01-01',
                'first_funding_at': f'{founded_year + 1}-01-01',
                'last_funding_at': f'{founded_year + 2}-01-01',
            }

            label, proba, classes = predict_single(row, art)

            st.markdown("---")
            st.subheader("Prediction Result")

            # Color-coded result
            color_map = {
                'operating': '🟢', 'acquired': '🔵',
                'closed': '🔴', 'ipo': '🟡',
            }
            emoji = color_map.get(label, '⚪')
            st.markdown(f"### {emoji} Predicted Status: **{label.upper()}**")

            # Probability bars
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                'Status': classes,
                'Probability': proba,
            }).sort_values('Probability', ascending=True)

            st.bar_chart(prob_df.set_index('Status'), horizontal=True)

            # Confidence warning
            max_prob = proba.max()
            if max_prob < 0.6:
                st.warning(
                    f"⚠️ Low confidence ({max_prob:.1%}). "
                    "The model is uncertain — human review recommended."
                )

    # ────────────────────────────────────────────
    # TAB 2: Batch Prediction
    # ────────────────────────────────────────────
    with tab2:
        st.markdown(
            "Upload a CSV file with the same columns as the training data. "
            "The model will predict the status for each startup."
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df_upload):,} rows")
                st.dataframe(df_upload.head())

                if st.button("🔮 Predict Batch", type="primary"):
                    with st.spinner("Running predictions..."):
                        results = predict_batch(df_upload, art)

                    st.success(f"Predictions complete for {len(results):,} startups!")
                    st.dataframe(results)

                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv, "startup_predictions.csv",
                        "text/csv",
                    )

                    # Summary
                    st.subheader("Prediction Summary")
                    summary = results['predicted_status'].value_counts()
                    st.bar_chart(summary)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "*PFA 2025–2026 · ENSAM-Rabat · Classification de Startups: DL vs ML*  \n"
        "*Model: XGBoost (best Macro F1) · Pipeline: Géron-style sklearn ColumnTransformer*"
    )


if __name__ == "__main__":
    main()
