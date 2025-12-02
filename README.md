# A gas sensor based on free-standing SWCNT film for selective recognition of toxic and flammable gases under thermal cycling protocols
Contains files, notebooks and source raw data used to develop and train the models described in the article [10.1016/j.snb.2024.136116](https://linkinghub.elsevier.com/retrieve/pii/S0925400524008463)

---
**Repository filling in progress**
---


# 🌡️ SWCNT Gas Sensor Pattern Recognition via Thermocycling

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimization-purple)](https://optuna.org/)

**A production-ready machine learning pipeline for gas classification and concentration prediction from SWCNT sensor time-series data.**

---

## 🎯 Project Highlights

- **Multi-model ML pipeline**: CatBoost (classification/regression) + LSTM (multi-output regression)
- **Experiment tracking**: Full MLflow integration with 100+ Optuna trials
- **Time-series handling**: Custom train/test splitting to prevent data leakage
- **Feature engineering**: PCA-based dimensionality reduction, dedrifting preprocessing
- **Reproducibility**: Docker support, version-controlled configs

---

## 📊 Problem Statement

Single-walled carbon nanotube (SWCNT) gas sensors generate noisy time-series data during **thermocycling** (402 datapoints/cycle). The challenge:
1. Classify gas type (NO₂, H₂S, Acetone) from sensor response patterns
2. Predict gas concentration (10, 15, 25 ppm) with multi-output regression
3. Handle drift, noise, and temporal dependencies

**Solution**: Hybrid ML approach with CatBoost for tabular features + LSTM for sequential patterns.

---

## 🏆 Results

| Model | Task | Metric | Performance |
|-------|------|--------|------------|
| CatBoost | Gas Classification | F1 Score (macro) | **0.91** |
| CatBoost | Concentration Regression | RMSE (ppm) | **4.2** |
| LSTM | Multi-output Regression | RMSE (ppm) | **3.6** |

*Validated via time-series cross-validation (8-fold split by measurement cycle)*

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/bahrs/gas-sensor_SWCNT_film-data-classification
cd gas-sensor_SWCNT_film-data-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
python scripts/run_preprocessing.py
```

### 4. Train Models
```bash
# CatBoost classification
python scripts/train_catboost.py --task classification

# LSTM multi-output regression
python scripts/train_lstm.py --epochs 150 --batch_size 128
```

### 5. Launch MLflow UI
```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000 to view experiments
```

---

## 📁 Repository Structure
```
swcnt-gas-sensor-ml/
├── notebooks/           # Jupyter demos (EDA, training, viz)
├── src/                 # Core modules (preprocessing, models, evaluation)
├── scripts/             # Standalone training/optimization scripts
├── configs/             # YAML configs for reproducibility
├── data/                # Raw + processed data
└── docs/                # Methodology, results documentation
```

---

## 🔬 Technical Approach

### Preprocessing Pipeline
1. **Dedrifting**: Savitzky-Golay or exponential smoothing across voltage envelopes
2. **Cycle reshaping**: 402-point time series → tabular features per cycle
3. **Feature engineering**: PCA (15-150 components, optimized via Optuna)

### Models
- **CatBoost**: Gradient boosting for classification + regression tasks
- **LSTM**: Recurrent neural network for sequential pattern learning

### Hyperparameter Optimization
- **Tool**: Optuna with MedianPruner
- **Trials**: 1000+ runs (8 hours)
- **Tracking**: All experiments logged to MLflow

### Validation Strategy
Time-series split to prevent lookahead bias:
- Train on cycles 1-7 → Test on cycle 8
- Train on cycles 1-8 → Test on cycle 9
- ...

---

## 📈 Visualizations

See `notebooks/05_visualization.ipynb` for:
- Thermocycling protocol diagrams
- Response/recovery time analysis
- Optuna optimization history
- Confusion matrices + calibration curves

---

## 🛠️ Technologies

**Core Stack**:
- Python 3.13
- scikit-learn, pandas, NumPy

**ML Frameworks**:
- CatBoost (gradient boosting)
- TensorFlow/Keras (LSTM)

**MLOps**:
- MLflow (experiment tracking)
- Optuna (hyperparameter tuning)
- Docker (containerization)

**Visualization**:
- Plotly (interactive plots)
- Matplotlib/Seaborn

---

## 📚 Documentation

- **Methodology**: [docs/methodology.md](docs/methodology.md)
- **Results Summary**: [docs/results.md](docs/results.md)
- **Data Description**: [data/README.md](data/README.md)

---

## 🤝 About This Project

This project demonstrates:
- ✅ Production-grade ML pipeline design
- ✅ Experiment tracking and reproducibility
- ✅ Time-series best practices (no data leakage)
- ✅ Hyperparameter optimization at scale
- ✅ Clean, modular code architecture

**Built as part of PhD research, refined as a data science portfolio project.**

---

## 📧 Contact

**Konstantin Zamansky** [ORCID](https://orcid.org/0009-0005-6495-1985) | [LinkedIn](https://www.linkedin.com/in/konstantin-zamansky-244837354/)

---

## 📄 License

GPL-3.0 License - see [LICENSE](LICENSE) for details

