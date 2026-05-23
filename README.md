# TEIP-GBM Vaccine Pipeline

**TEIP** (Tumor Epitope Immunogenicity Pipeline) is a research-use computational framework for prioritizing candidate peptide-HLA targets for glioblastoma (GBM) vaccine development, with emphasis on OIP5-derived tumor-associated epitopes.

This repository contains a reproducible prototype implementation including peptide/HLA encoding, tumor-context feature handling, dual-output neural scoring, batch prediction scripts, a Streamlit public UI scaffold, example input/output files, and documentation.

> **Research-use only.** TEIP is not clinically validated and must not be used for medical diagnosis, treatment selection, or patient care.

## Quick start

```bash
git clone https://github.com/anishsundargs/teip-gbm-vaccine-pipeline.git
cd teip-gbm-vaccine-pipeline
pip install -r requirements.txt
python scripts/predict_batch.py --input data/example_input.csv --output outputs/example_predictions.csv --config configs/config.yaml --allow_fallback
streamlit run app/streamlit_app.py
```

## Repository structure

```text
teip/                 Core Python package
scripts/              Training and batch prediction scripts
app/                  Streamlit public UI
configs/              YAML configuration files
data/                 Small example data only
models/               Placeholder for trained checkpoints
docs/                 Method, schema, and limitation notes
analysis/             Cleaned analysis scripts
figures/              Generated figures
notebooks/            Cleaned notebooks
```

## Input format

Required prediction columns:

```text
peptide,hla_pseudo,oip5_tpm,b2m_tpm,tap1_tpm,tap2_tpm,tapbp_tpm,hla_a_tpm,hla_b_tpm,hla_c_tpm,ifng_score,ap_score
```

See `docs/input_schema.md`.

## Model checkpoint

Place a trained checkpoint at:

```text
models/teip_checkpoint.ckpt
```

Large checkpoints should not be committed directly unless using Git LFS or release assets.

## Limitations

TEIP ranks computational candidates. It does not prove natural antigen presentation, T-cell activation, vaccine efficacy, clinical benefit, or patient safety. All candidates require experimental validation.
