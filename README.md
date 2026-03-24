# Hybrid QML Optimization Benchmark

This repository contains code for benchmarking gradient-based, gradient-free, and hybrid optimization methods for hybrid quantum machine learning under matched quantum-evaluation budgets.

The suite includes implementations of:
- gradient-based and hybrid optimizers,
- gradient-free optimizers,
- shared helper classes for hybrid QML models and quantum layers, and
- experiment scripts for running benchmark studies across optimizer families.

---

## Repository Structure

- `code/`
  - `gb_hybrid_optimizers/`
    - `adam_sgd.py` - gradient-based baseline optimizers
    - `gls_adam_sgd.py` - hybrid / guided line-search variants
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
    - `abstract_classes.py` - shared abstract interfaces
    - `myquantumlayer.py` - quantum-layer implementation
    - `wrapper_hybrid_qml_model.py` - wrapper for hybrid QML models
  - `test_optimizers_GB_Hybird.py` - benchmark script for gradient-based / hybrid optimizers
  - `test_optimizers_GF.py` - benchmark script for gradient-free optimizers
- `data/`
  - stores benchmark data and experiment outputs
- `requirements/`
  - `requirements.txt` - dependencies for setting up the Python environment

---

## Installation

It is strongly recommended to create a virtual environment before installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements/requirements.txt
