# DiSOL for Geometry-Dependent Advection–Diffusion Equation

This folder contains the implementation of **Discrete Solution Operator Learning (DiSOL)** for solving geometry-dependent **advection–diffusion equations** on embedded Cartesian grids. The code supports **training**, **evaluation**, and **out-of-distribution (OOD)** testing under geometric variations and transport-dominated regimes.

This case corresponds to the advection–diffusion benchmarks reported in the DiSOL manuscript.

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
│   ├── Training_Advection_Diffusion_High_Pe.pt   # training dataset
│   └── OOD_Test_Advection_Diffusion.pt           # OOD test dataset
│
├── model_ckpt/
│   ├── ckpt_High_Pe_epoch_0300.pth               # example pretrained checkpoint
│   └── model_cfg.json                            # saved model configuration
│
├── train_DiSOL_Advection_Diffusion.py            # training script
├── run_DiSOL_Advection_Diffusion.py              # evaluation / testing script
├── DiSOL.py                                     # DiSOL model definition
├── dataset.py                                  # dataset utilities
├── requirements.txt
└── README.md
```

> **Note**  
> All training and OOD datasets are expected to be located in the local `./data` directory.

---

## 3. Dataset Preparation

Before training or testing, make sure all dataset files are placed in the `./data` directory.

### Training data
- **File name**: `Training_Advection_Diffusion_High_Pe.pt`
- **Location**: `./data/Training_Advection_Diffusion_High_Pe.pt`

### OOD test data
- **File name**: `OOD_Test_Advection_Diffusion.pt`
- **Location**: `./data/OOD_Test_Advection_Diffusion.pt`

---

## 4. Training a DiSOL Model

To train a DiSOL model for the advection–diffusion equation, run the following command from the project root directory:

```bash
python train_DiSOL_Advection_Diffusion.py --pt_path ./data/Training_Advection_Diffusion_High_Pe.pt --save_dir ./model_ckpt --epochs 300 --batch_size 200 --lr 1e-4 --weight_decay 1e-5 --save_every 10 --save_best
```

### Key arguments
- `--pt_path`: Path to the training dataset  
- `--save_dir`: Directory to store model checkpoints  
- `--epochs`: Number of training epochs  
- `--batch_size`: Training batch size  
- `--lr`: Learning rate  
- `--weight_decay`: Weight decay for optimizer  

---

## 5. Testing and Evaluation

To evaluate a trained DiSOL model and generate visualizations and metrics, run:

```bash
python run_DiSOL_Advection_Diffusion.py --pt_path ./data/Training_Advection_Diffusion_High_Pe.pt --ckpt ./model_ckpt/ckpt_High_Pe_epoch_0300.pth --model_cfg_json ./model_ckpt/model_cfg.json --out_dir ./output_results --vis_n 4 --ood_pt_path ./data/OOD_Test_Advection_Diffusion.pt --use_mask_metrics --apply_mask_to_pred
```

### Evaluation features
- Masked error metrics on the active computational domain  
- Gradient-based error metrics  
- OOD generalization testing under geometric shifts  
- Visualization of prediction fields and errors  

---

## 6. Output

All evaluation results will be written to:

```text
./output_results/
```

This directory includes:
- Saved prediction tensors
- Visualization figures
- Computed scalar metrics (including gradient-based metrics)

---

## 7. Notes on Reproducibility

- This case focuses on **transport-dominated (high Péclet number)** advection–diffusion problems.
- All models are trained and evaluated on normalized solution patterns with identical geometric masking.
- Geometry variations affect boundary activation and effective connectivity, making this case particularly challenging for continuous operator baselines.
