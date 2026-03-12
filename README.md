# Strabismus Diagnosis Analysis Pipeline

Reproducible Python implementation of the analyses reported in:

**Takinami S, Morita Y, Ito Y, Seita J, Oshika T.** *Exploration of angle thresholds for horizontal strabismus diagnosis based on receiver operating characteristic curve analysis and machine learning.* PLOS ONE. (under review)

**Repository:** https://github.com/namihira33/strabismus-analysis-pipeline

## Overview

This repository contains the analysis code for determining diagnostic thresholds for horizontal strabismus based on quantitative analysis of strabismus angle measurements. The pipeline includes:

- **ROC threshold analysis** with Youden Index optimization for near (33 cm) and distance (5 m) strabismus angles
- **Machine learning classification** using Logistic Regression (LR) and Light Gradient Boosting Machine (LGBM)
- **Bootstrap stability analysis** (n=1,000) for threshold confidence intervals
- **PPV/NPV estimation** using Bayes' theorem at realistic screening prevalence (2–3%)

All analyses use 5-fold cross-validation for performance evaluation.

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

### Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python strabismus_analysis_pipeline.py
```

**Note:** The clinical dataset (`Strabismus_tknmAndIto_enhanced.csv`) is not included in this repository due to patient privacy restrictions. See **Data Availability** below.

## File Structure

```
├── strabismus_analysis_pipeline.py   # Main analysis script
├── requirements.txt                  # Python package dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License
```

## Data Availability

The clinical data used in this study cannot be shared publicly due to ethical restrictions related to patient privacy. Data access requests can be directed to:

**Tsukuba Clinical Research & Development Organization (T-CReDO)**
Clinical Research Support Center, Review Committee Office
University of Tsukuba
1-1-1 Tennodai, Tsukuba, Ibaraki 305-8575, Japan
Email: tcr.nintei@un.tsukuba.ac.jp
Phone: +81-29-853-3749

## Input Data Format

The script expects a CSV file with the following columns:

| Column Name | Description |
|---|---|
| `水平斜視角 近見` | Near horizontal strabismus angle (prism diopters, Δ) |
| `水平斜視角 遠見` | Distance horizontal strabismus angle (prism diopters, Δ) |
| `斜視か` | Diagnostic label (1 = strabismus, 0 = normal) |

## Notes

- The original near/distance single-measurement ROC analyses were implemented in JavaScript without fixed seeds.
- The original LGBM analysis was implemented using the `StrabismusMLPipeline` class in Python.
- This script provides a unified, reproducible Python implementation.
- Minor numerical differences from reported values may occur due to differences in cross-validation split implementation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite the associated manuscript:

```
Takinami S, Morita Y, Ito Y, Seita J, Oshika T. Exploration of angle thresholds
for horizontal strabismus diagnosis based on receiver operating characteristic
curve analysis and machine learning. PLOS ONE. (under review)
https://github.com/namihira33/strabismus-analysis-pipeline
```
