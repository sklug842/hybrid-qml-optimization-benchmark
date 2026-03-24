# Hybrid QML Optimization Benchmark

Code accompanying the manuscript:

**Stefan Klug, Maximilian Moll**  
*Gradient-Based versus Gradient-Free Optimization in Hybrid Quantum Machine Learning: A Systematic Benchmark*

## Overview

This repository contains code for benchmarking gradient-based, gradient-free, and hybrid optimization methods for hybrid quantum machine learning under matched quantum-evaluation budgets.

The benchmark suite includes:

- gradient-based optimizers,
- gradient-free optimizers,
- hybrid optimization strategies,
- shared helper classes for hybrid quantum-classical models and quantum layers, and
- experiment scripts for running the benchmark across optimizer families.

## Repository Structure

- `code/`
  - `gb_hybrid_optimizers/`
    - `adam_sgd.py` – gradient-based baseline optimizers
    - `gls_adam_sgd.py` – hybrid / guided line-search variants
  - `gf_optimizers/`
    - `aro.py`
    - `bb_bc.py`
    - `cem.py`
    - `ga_elite.py`
    - `ga_tourn.py`
    - `gwo.py`
    - `nes.py`
    - `pso_bgb.py`
    - `spo.py`
  - `help_classes/`
    - `abstract_classes.py` – shared abstract interfaces
    - `myquantumlayer.py` – quantum-layer implementation
    - `wrapper_hybrid_qml_model.py` – wrapper for hybrid QML models
  - `test_optimizers_GB_Hybird.py` – benchmark script for gradient-based / hybrid optimizers
  - `test_optimizers_GF.py` – benchmark script for gradient-free optimizers
- `data/`
  - stores benchmark data and experiment outputs
- `requirements/`
  - `requirements.txt` – dependencies for setting up the Python environment

## Installation

It is recommended to create a virtual environment before installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements/requirements.txt
