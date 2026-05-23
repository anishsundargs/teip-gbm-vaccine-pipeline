from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from teip.datamodule import REQUIRED_COLUMNS
from scripts.predict_batch import fallback_score, run_model
from teip.config import load_config


st.set_page_config(
    page_title="TEIP GBM Vaccine Target Prioritization",
    page_icon="🧬",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
    }

    .hero h1 {
        font-size: 2.45rem;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        max-width: 850px;
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        margin-right: 0.45rem;
        margin-top: 0.8rem;
        font-size: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    .card {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .card h3 {
        margin-top: 0;
        color: #0f172a;
        font-size: 1.05rem;
    }

    .card p {
        color: #475569;
        font-size: 0.92rem;
    }

    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        margin-bottom: 1.25rem;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>TEIP-GBM Vaccine Target Prioritization</h1>
        <p>
        A research-use computational interface for ranking candidate peptide-HLA targets
        in glioblastoma vaccine-development workflows, with emphasis on OIP5-derived
        tumor-associated epitopes.
        </p>
        <span class="pill">Glioblastoma</span>
        <span class="pill">Peptide-HLA scoring</span>
        <span class="pill">OIP5</span>
        <span class="pill">Bioinformatics pipeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
    <b>Research-use only.</b> TEIP is not clinically validated. It does not diagnose disease,
    prove vaccine efficacy, or make treatment recommendations. All ranked candidates require
    experimental validation.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Info cards
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
            <h3>1. Upload candidates</h3>
            <p>Provide a CSV containing peptide sequences, HLA pseudo-sequences, and tumor-context features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <h3>2. Score peptide-HLA rows</h3>
            <p>The interface checks the input schema and generates ranked presentation and targetability scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
            <h3>3. Export ranked output</h3>
            <p>Download a prediction table for downstream epitope prioritization and experimental planning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("TEIP")
    st.markdown("**Tumor Epitope Immunogenicity Pipeline**")
    st.markdown("---")
    st.markdown("### Required input columns")
    st.code("\n".join(REQUIRED_COLUMNS))
    st.markdown("---")
    st.markdown(
        """
        **Links**

        - GitHub repository:  
          https://github.com/anishsundargs/teip-gbm-vaccine-pipeline
        """
    )


# -----------------------------
# Upload section
# -----------------------------
st.subheader("Run candidate scoring")

left, right = st.columns([1.1, 1])

with left:
    uploaded = st.file_uploader(
        "Upload candidate peptide-HLA CSV",
        type=["csv"],
        help="Upload a CSV with the required TEIP input columns.",
    )

with right:
    example_path = ROOT / "data" / "example_input.csv"
    if example_path.exists():
        example_df = pd.read_csv(example_path)
        st.markdown("Use the included example file to test the interface.")
        st.download_button(
            "Download example input CSV",
            example_df.to_csv(index=False),
            file_name="example_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

with st.expander("View required input schema"):
    schema_df = pd.DataFrame(
        {
            "Column": REQUIRED_COLUMNS,
            "Status": ["Required"] * len(REQUIRED_COLUMNS),
        }
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)


if uploaded is None:
    st.info("Upload a CSV file or download the example input file to test the app.")
    st.stop()


# -----------------------------
# Input validation
# -----------------------------
df = pd.read_csv(uploaded)

st.subheader("Input preview")
st.dataframe(df.head(20), use_container_width=True)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Rows uploaded", len(df))
m2.metric("Required columns present", f"{len(REQUIRED_COLUMNS)}/{len(REQUIRED_COLUMNS)}")
m3.metric("Unique peptides", df["peptide"].nunique() if "peptide" in df.columns else "N/A")

cfg = load_config("configs/config.yaml")
checkpoint = Path("models/teip_checkpoint.ckpt")

st.divider()

run_clicked = st.button("Run TEIP scoring", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Scoring candidate peptide-HLA rows..."):
        if checkpoint.exists():
            result = run_model(df, cfg, str(checkpoint))
            st.success("Predictions generated using trained checkpoint.")
        else:
            result = fallback_score(df)
            st.warning(
                "No trained TEIP checkpoint is currently deployed. "
                "These are fallback demo scores for interface testing only, "
                "not model-based biological predictions."
            )

    result = result.sort_values("teip_score", ascending=False)

    st.subheader("Ranked candidates")

    top1, top2, top3 = st.columns(3)
    top1.metric("Top TEIP score", f"{result['teip_score'].max():.3f}")
    top2.metric("Mean TEIP score", f"{result['teip_score'].mean():.3f}")
    top3.metric("Candidates scored", len(result))

    display_cols = [
        c
        for c in [
            "peptide",
            "hla_allele",
            "pred_presentation",
            "pred_targetability",
            "teip_score",
            "mode",
        ]
        if c in result.columns
    ]

    st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download ranked predictions",
        result.to_csv(index=False),
        file_name="teip_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if "peptide" in result.columns and "teip_score" in result.columns:
        st.subheader("Top candidate score chart")
        chart_df = result.head(20).set_index("peptide")[["teip_score"]]
        st.bar_chart(chart_df)

    with st.expander("Full output table"):
        st.dataframe(result, use_container_width=True)
