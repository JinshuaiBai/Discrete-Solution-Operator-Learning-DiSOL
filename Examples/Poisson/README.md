# DiSOL for Geometry-Dependent Poisson Equation

This folder contains the implementation of **Discrete Solution Operator Learning (DiSOL)** for solving geometry-dependent Poisson equations on embedded Cartesian grids. The code supports **training**, **evaluation**, and **out-of-distribution (OOD)** testing under geometric variations, and is designed to be fully controllable via command-line arguments.

This case corresponds to the Poisson benchmark reported in the DiSOL manuscript.

---

## 1. Environment Setup

### 1.1 Tested Environment

The code has been tested with the following environment:

- **Python**: 3.13  
- **PyTorch**: 2.8.0 + CUDA 12.8  
- **CUDA**: 12.8  
- **cuDNN**: 9.1  
- **GPU**: NVIDIA GPU with CUDA support

### 1.2 Required Python Packages

The minimal runtime dependencies are listed below:

```txt
numpy==2.1.2
tqdm==4.67.1
matplotlib==3.10.6
torch==2.8.0
```

> **Note on PyTorch installation**  
> The installed PyTorch build uses CUDA 12.8 (`torch==2.8.0+cu128`).  
> It is recommended to install PyTorch explicitly following the official instructions at:  
> https://pytorch.org/get-started/locally/

Example (pip):
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 2. Directory Structure

```text
.
├── data/
│   ├── Training_Poisson.pt        # training dataset
│   └── OOD_Test_Poisson.pt        # OOD test dataset
│
├── model_ckpt/
│   ├── ckpt_epoch_0500.pth        # example pretrained checkpoint
│   └── model_cfg.json             # saved model configuration
│
├── train_DiSOL_Poisson.py         # training script
├── run_DiSOL_Poisson.py           # evaluation / testing script
├── DiSOL.py                       # DiSOL model definition
├── dataset.py                     # dataset utilities
├── requirements.txt
└── README.md
```

---

## 3. Dataset Preparation

Before training or testing, the dataset files must be placed in the `./data` directory.

### Training data
- **File name**: `Training_Poisson.pt`
- **Location**: `./data/Training_Poisson.pt`

> The download link for the training dataset will be provided separately.

### OOD test data
- **File name**: `OOD_Test_Poisson.pt`
- **Location**: `./data/OOD_Test_Poisson.pt`

---

## 4. Training a DiSOL Model

To train a DiSOL model for the Poisson equation, run the following command from the project root directory:

```bash
python train_DiSOL_Poisson.py --pt_path ./data/Training_Poisson.pt --save_dir ./model_ckpt --epochs 500 --batch_size 200 --lr 1e-4 --weight_decay 1e-5 --save_every 10 --save_best
```

---

## 5. Testing and Evaluation

To evaluate a trained DiSOL model and generate visualizations and metrics, run:

```bash
python run_DiSOL_Poisson.py --pt_path ./data/Training_Poisson.pt --ckpt ./model_ckpt/ckpt_epoch_0500.pth --model_cfg_json ./model_ckpt/model_cfg.json --out_dir ./output_results --vis_n 4 --ood_pt_path ./data/OOD_Test_Poisson.pt --use_mask_metrics --apply_mask_to_pred --compute_grad_metrics --dump_predictions
```

---

## 6. Output

All evaluation results will be written to:

```text
./output_results/
```

---

## 7. Notes on Reproducibility

- All models in this case are trained and evaluated on **normalized solution patterns**, with identical masking applied to the active computational domain.
- Absolute-amplitude recovery (if required for downstream applications) is treated as a post-processing step.
- Random seeds and training protocols are consistent across baseline comparisons.
