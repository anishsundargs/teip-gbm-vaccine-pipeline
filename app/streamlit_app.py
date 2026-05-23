from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from teip.datamodule import REQUIRED_COLUMNS
from scripts.predict_batch import fallback_score, run_model
from teip.config import load_config

st.set_page_config(page_title="TEIP GBM Vaccine Target Prioritization", layout="wide")
st.title("TEIP: GBM Vaccine Target Prioritization")
st.caption("Research-use peptide-HLA prioritization. Not clinically validated.")
st.warning("TEIP is for research use only. It does not diagnose disease, prove vaccine efficacy, or make clinical treatment recommendations.")
uploaded = st.file_uploader("Upload candidate peptide-HLA CSV", type=["csv"])
example_path = ROOT / "data" / "example_input.csv"
if example_path.exists():
    example_df = pd.read_csv(example_path)
    st.download_button(
        "Download example input CSV",
        example_df.to_csv(index=False),
        file_name="example_input.csv",
        mime="text/csv",
    )
with st.expander("Required columns"):
    st.code("\n".join(REQUIRED_COLUMNS))
if uploaded is None:
    st.info("Upload a CSV or use the example file in data/example_input.csv."); st.stop()
df = pd.read_csv(uploaded)
st.subheader("Input preview"); st.dataframe(df.head(), use_container_width=True)
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}"); st.stop()
cfg = load_config("configs/config.yaml"); checkpoint = Path("models/teip_checkpoint.ckpt")
if st.button("Run TEIP scoring"):
    if checkpoint.exists():
        result = run_model(df, cfg, str(checkpoint)); st.success("Predictions generated using trained checkpoint.")
    else:
        result = fallback_score(df); st.warning("No trained checkpoint found. Showing fallback demo scores only.")
    result = result.sort_values("teip_score", ascending=False)
    st.subheader("Ranked candidates"); st.dataframe(result, use_container_width=True)
    st.download_button("Download predictions", result.to_csv(index=False), file_name="teip_predictions.csv", mime="text/csv")
    if "peptide" in result.columns: st.bar_chart(result.head(20).set_index("peptide")[["teip_score"]])
