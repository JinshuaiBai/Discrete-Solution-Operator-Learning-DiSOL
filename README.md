# Discrete Solution Operator Learning (DiSOL)

Official implementation of **DiSOL (Discrete Solution Operator Learning)** for **geometry-dependent PDEs**, introduced in:

- **Paper (arXiv):** Discrete Solution Operator Learning for Geometry-Dependent PDEs — https://arxiv.org/abs/2601.09143  
- **Datasets (Zenodo):** https://doi.org/10.5281/zenodo.18639633

DiSOL targets **geometry-driven discrete structural variation** (e.g., topology changes, boundary-type activation, and changes of active computational regions) by learning a **discrete, procedure-level** solution operator on embedded grids, rather than approximating a single smooth mapping between continuous function spaces.

---

## What’s in this repository

- End-to-end PyTorch training / validation / testing code for **four benchmark cases** (in the paper order):
  1. Poisson
  2. Advection Diffusion
  3. Linear Elasticity
  4. Thermal Conduction
- **Pretrained checkpoints** and **OOD evaluation** utilities (see each case folder).

> **Code Ocean capsule:** an executable Code Ocean capsule for peer review / reproducibility is provided: https://codeocean.com/capsule/4396603/tree/v1 

---

## Repository structure

```
.
├── Examples/                 # PyTorch implementation + per-case scripts/checkpoints
│   ├── Poisson/              # (case-specific README inside)
│   ├── Advection_Diffusion/
│   ├── Elasticity/
│   └── Thermal_Conduction/
└── data_generation/          # MATLAB scripts for dataset generation
```

Each case folder under `Examples/` contains:
- training and validation scripts
- OOD test data or instructions to obtain it
- pretrained model(s) (or download instructions)
- a case-specific `README.md` with the exact commands

---

## Quick start (recommended workflow)

1) **Install dependencies** (PyTorch + standard scientific Python stack).  
2) Go to a case folder under `Examples/` and follow its README.

Example:
```bash
cd Examples/Poisson
# then follow the commands in Examples/Poisson/README.md
```

> **GPU note:** full-scale training in the paper used high-memory GPUs (≈40GB VRAM).  
> For limited GPUs, reduce batch size / epochs as needed (the case READMEs provide practical settings).

---

## Datasets

All datasets used in this work are publicly available on Zenodo:

- **Zenodo DOI:** https://doi.org/10.5281/zenodo.18639634

The Zenodo record contains the **full datasets** used for the paper experiments.  
For lightweight verification (e.g., Code Ocean), toy subsets may be used; see the capsule (when released) or the per-case README instructions.

---

## Reproducibility notes

- **Paper environment:** PyTorch **2.8.0**
- **Compatibility:** the code may run on other PyTorch versions, but small numerical differences can occur.  
  If you encounter version-specific issues, please open an issue with your `python/torch/cuda` versions and a minimal log.

---

## Citation

If you use this repository or the datasets, please cite:

```bibtex
@article{bai2026disol,
  title   = {Discrete Solution Operator Learning for Geometry-Dependent PDEs},
  author  = {Bai, Jinshuai and Li, Haolin and Sharif Khodaei, Zahra and Aliabadi, M. H. and Gu, YuanTong and Feng, Xi-Qiao},
  journal = {arXiv preprint arXiv:2601.09143},
  year    = {2026}
}

@dataset{ZenodoDataset,
  author = {Bai, Jinshuai and Li, Haolin and Sharif Khodaei, Zahra and Aliabadi, M.H. and Gu, YuanTong and Feng, Xi-Qiao},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18639634},
  url = {https://doi.org/10.5281/zenodo.18639634}
}
```

---

## License / contact

If you have questions (or find a bug), please open a GitHub issue with:
- the case name (Poisson / Advection–Diffusion / Elasticity / Thermal)
- your environment (OS, Python, PyTorch, CUDA)
- the command you ran and the full error log
