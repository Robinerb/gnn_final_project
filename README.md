# Inverse Molecular Design using Multitask Variational Autoencoders

**Author:** Robin Erb | **Program:** Data and Computer Science, Heidelberg University  

## Project Overview
This repository contains the code for a comparative study of SMILES and SELFIES molecular representations within a multitask sequence-to-sequence Variational Autoencoder (VAE). The project navigates the continuous chemical latent space to optimize the Quantitative Estimate of Drug-likeness (QED), diagnosing optimization pathologies like posterior collapse and reward hacking in limited-data regimes.

## Repository Structure
```text
.
├── data/                       # ZINC dataset
├── checkpoints/                # Saved PyTorch model weights (.pth)
├── results/                    # Output CSVs (training history, optimization)
├── figures/                    # Generated plots and visualizations
├── model.py & model_qed.py     # Base VAE and Multitask VAE architectures
├── download_data.py            # Fetches ZINC data
├── preprocess.py               # Tokenization and zero-padding
├── train.py & train_qed.py     # Training loops for standard and multitask VAEs
├── evaluation.py & validate.py # Evaluates generative validity and robustness
├── latent_walk.py              # Latent space interpolations
├── optimize_qed.py             # Latent space gradient ascent for inverse design
├── plot_*.py                   # Scripts for generating loss curves and bar charts
├── run_final_eval.py           # Master script for the full evaluation suite
└── README.md
```

## Quick Start Pipeline
### 1. Prepare Data & Train Models:
- python download_data.py && python preprocess.py
- python train.py 
- python train_qed.py

### 2. Evaluate & Analyze Latent Space:
- python run_final_eval.py
- python latent_walk.py
- python optimize_qed.py

### 3. Generate Visualizations:
- python plot_history.py
- python plot_final_result.py

## Key Results
- Robustness: SELFIES achieved 100% chemical validity under random sampling; SMILES failed at 0%.
- Inverse Design: Gradient ascent successfully maximized the internal QED predictor (~0.98), but RDKit validation showed true scores of ~0.05, exposing extreme reward hacking via infinite alkane chains.