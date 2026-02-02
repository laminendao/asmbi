[![GitHub Repo](https://img.shields.io/badge/GitHub-IDFC-blue?logo=github)](https://github.com/laminendao/asmbi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Interpretable Divisive Feature Clustering (IDFC)

This repository provides the code used in the paper:

> **Explainable Remaining Useful Life Prediction Using an Interpretable Divisive Feature Clustering**  
> Mouhamadou Lamine Ndao, Genane Youness, Ndeye Niang, Gilbert Saporta  
> *Applied Stochastic Models in Business and Industry* (under revision)  
> DOI: forthcoming

The repository contains the **implementation, experiments, and evaluation pipeline** used to produce all numerical results reported in the paper.

---

## What is IDFC?

**Interpretable Divisive Feature Clustering (IDFC)** is an unsupervised dimensionality reduction method designed for **high-dimensional and highly correlated sensor data**, with a particular focus on **explainable Remaining Useful Life (RUL) prediction**.

Unlike classical dimensionality reduction methods based on latent components (e.g. PCA), IDFC:
- clusters **original input features**,
- selects **real and physically interpretable sensors** as cluster representatives,
- explicitly identifies **atypical features** that do not conform to dominant correlation structures.

IDFC is specifically designed to improve the **reliability of post-hoc explainability methods**, such as SHAP, in the presence of multicollinearity.

---

## Methodological overview

IDFC combines ideas from existing feature clustering approaches:

- **VARCLUS** (divisive hierarchical clustering) for initialization,
- **CLV (Clustering of Variables)** for refinement of feature groups,
- a **K+1 strategy** (Vigneau, 2016) to isolate atypical features,
- selection of one **representative original feature per cluster**, instead of latent components.

This structure allows dimensionality reduction while preserving semantic meaning and interpretability.

---

## Repository structure

```
asmbi/
├── Rfiles/                 # R implementations of CLV-related procedures
├── data/                   # NASA C-MAPSS datasets
├── utils/                  # preprocessing, scoring, and helper functions
├── stability_folder/       # computation of stability metrics
└── notebooks/              # Jupyter notebooks for experiments and figures
```

---

## What this repository contains

✔ IDFC feature clustering and selection  
✔ Preprocessing pipeline (normalization, exponential smoothing, sliding windows)  
✔ LSTM-based RUL prediction model  
✔ SHAP-based explainability analysis  
✔ Evaluation of explanation quality using:
- coherence
- stability
- acumen
- computation time  

⚠️ This repository **does not include** LIME or L2X experiments, which are **not part of the paper**.

---

## Dataset

The experiments are conducted on the **NASA C-MAPSS** dataset for turbofan engine prognostics:
- FD001, FD002, FD003, FD004 subsets
- multivariate time series with multiple operating conditions and fault modes

The dataset is publicly available and included here for reproducibility.

---

## References

- Ndao M.L. et al. – *Explainable Remaining Useful Life Prediction Using an Interpretable Divisive Feature Clustering* (under review)
- Vigneau E. (2016). *Dimensionality reduction by clustering of variables while setting aside atypical variables*
- Vigneau & Qannari (2003). *Clustering of variables around latent components*
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*
- Saxena et al. (2008). *C-MAPSS dataset*

---

## Authors

Developed by  
**M. L. Ndao**, G. Youness, N. Niang, G. Saporta  
Affiliations: CESI LINEACT / CNAM-CEDRIC  
Year: 2025

---

## Citation

If you use this code, please cite the associated paper and include a link to this repository:

```
@article{Ndao2025IDFC,
  title={Explainable Remaining Useful Life Prediction Using an Interpretable Divisive Feature Clustering},
  author={Ndao, Mouhamadou Lamine and Youness, Genane and Niang, Ndeye and Saporta, Gilbert},
  journal={Applied Stochastic Models in Business and Industry},
  year={2025}
}
```

---

## License

This project is released under the MIT License.
