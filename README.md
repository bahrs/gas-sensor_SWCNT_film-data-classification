# 🌡️ SWCNT Gas Sensor Pattern Recognition via Thermocycling

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimization-purple)](https://optuna.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

**Production-ready ML pipeline for gas classification and concentration prediction from SWCNT sensor time-series data**

Published research: [10.1016/j.snb.2024.136116](https://linkinghub.elsevier.com/retrieve/pii/S0925400524008463)

---

## 🎯 Project Overview

This project demonstrates end-to-end machine learning pipeline development for a real-world sensor analysis problem. Single-walled carbon nanotube (SWCNT) gas sensors generate complex time-series signals during thermal cycling. The challenge is to extract meaningful patterns from noisy 402-point cycles to identify gas types and predict concentrations.

**Business Value**: Automated gas detection systems for environmental monitoring, industrial safety, and air quality control.

### Key Accomplishments

- **Multi-model architecture**: CatBoost gradient boosting + LSTM neural networks
- **Rigorous validation**: Time-series cross-validation preventing data leakage
- **Hyperparameter optimization**: 100+ Optuna trials with automated pruning
- **Experiment tracking**: Full MLflow integration for reproducibility
- **Production considerations**: Docker containerization, config-driven design, modular codebase

---

## 📊 Problem Statement

**Input**: Noisy 402-point time-series from SWCNT sensors during thermocycling  
**Outputs**:
1. **Gas classification**: Identify gas type (NO₂, H₂S, Acetone, or clean air)
2. **Concentration regression**: Predict gas concentration (10, 15, 25 ppm)

**Challenges**:
- Sensor drift over time
- High dimensionality (402 features per cycle)
- Temporal dependencies between measurements
- Limited labeled data (~3,500 cycles across 3 gases)

---

## 🏆 Results

| Model | Task | Metric | Performance |
|-------|------|--------|------------|
| **CatBoost** | Gas Classification | F1-macro | **0.91** |
| **CatBoost** | Concentration Regression | RMSE | **4.2 ppm** |
| **LSTM** | Multi-output Regression | RMSE | **2.2 ppm** |

*Validated via 8-fold time-series cross-validation with expanding window*

### Model Performance Highlights

- **Classification**: 91% F1-score distinguishing 4 classes (3 gases + air)
- **Regression**: Sub-5 ppm error on concentration prediction (10-25 ppm range)
- **Stability**: Low variance across CV folds indicating robust generalization
- **Efficiency**: CatBoost trains in <2 min; LSTM in <10 min per fold

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- 4GB+ RAM
- (Optional) Docker for containerized execution

## Installation

### Option A — Use as a library (recommended)
```bash
git clone https://github.com/bahrs/gas-sensor_SWCNT_film-data-classification.git
cd gas-sensor_SWCNT_film-data-classification
python -m venv .venv
```
# Linux/macOS:
```bash
source .venv/bin/activate
```
# Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```
```bash
pip install .
```
### Basic Usage

```bash
# 1. Run hyperparameter optimization (CatBoost regression example)
thermocycling optimize configs/config_lstm_regression.yaml

# 2. Launch MLflow UI to view results
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000

```

### Docker Deployment

```bash
# Build image
docker build -t gas-sensor-ml .

# Run optimization in container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/mlruns:/app/mlruns \
           -p 5050:5050 \
           gas-sensor-ml

# Access MLflow at http://localhost:5050
```

---

## 📁 Repository Structure

```
gas-sensor-ml/
├── src/thermocycling             # Core library code
│   ├── pipeline/                     # Data loading & assembly
│   │   ├── loading.py           # Load raw Parquet files
│   │   ├── cleaning.py          # Apply manual data trimming
│   │   ├── assemble.py          # Build full dataset with dedrifting
│   │   └── paths.py             # Centralized file paths
│   ├── preprocessing/            # Feature engineering
│   │   ├── smoothing.py         # Dedrifting algorithms (Exp, Savitzky-Golay)
│   │   └── train_test.py        # Time-series CV splitting + PCA
│   └── models/                   # Model definitions
│       ├── catboost_model.py    # CatBoost classifier/regressor
│       ├── lstm_model.py        # LSTM architecture
│       └── optuna_objectives.py # Optimization objectives + MLflow
├── configs/                      # YAML experiment configs
│   ├── config_lstm_regression.yaml
│   ├── config_catboost_classification.yaml
│   └── config_catboost_regression.yaml
├── scripts/                      # Executable scripts
│   └── run_optimization_from_config.py  # Main training script
├── notebooks/                    # Jupyter analysis notebooks
│   ├── Test_notebook.ipynb      # Setup validation
│   └── Optuna_tuning.ipynb      # Hyperparameter analysis
├── data/                         # Data directory (git-ignored)
│   ├── raw/                     # Original .brotli sensor files
│   └── processed/               # .parquet preprocessed data
├── docs/                         # Documentation
│   ├── methodology.md           # Technical methodology
│   └── DOCKER_SETUP.md          # Docker instructions
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔬 Technical Approach

### Data Pipeline

```
Raw Sensor Data (.brotli compressed)
    ↓
Cleaning & Cycle Segmentation (402 points/cycle)
    ↓
Dedrifting (Exponential or Savitzky-Golay smoothing)
    ↓
Feature Engineering (Optional PCA: 10-200 components)
    ↓
Time-Series CV Split (Expanding window)
    ↓
Model Training (CatBoost or LSTM)
    ↓
Evaluation & MLflow Logging
```

### Key Design Decisions

**1. Time-Series Cross-Validation**
- **Problem**: Standard k-fold CV causes data leakage with temporal data
- **Solution**: Expanding window split by measurement cycle
  - Fold 1: Train on cycles 1-7 → Test on cycle 8
  - Fold 2: Train on cycles 1-8 → Test on cycle 9
  - Ensures no future information leaks into training

**2. Dedrifting Preprocessing**
- **Problem**: Sensor baseline shifts over time
- **Solution**: Extract voltage envelope at cycle position 201, apply smoothing, subtract from raw signal
- **Options**: Exponential smoothing (α=0.0217) or Savitzky-Golay filter

**3. Dimensionality Reduction**
- **Problem**: 402 features per cycle → overfitting risk
- **Solution**: PCA tuned via Optuna (optimal: 20-150 components)
- **Benefit**: Reduces train time, improves generalization

**4. Hybrid Model Strategy**
- **CatBoost**: Fast, handles tabular data, no sequence modeling
- **LSTM**: Captures temporal patterns within gas-specific sequences
- **Trade-off**: LSTM achieves lower RMSE but requires more training time

### Hyperparameter Optimization

**Optuna Configuration**:
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner with warmup (prunes poor trials early)
- **Search Space**:
  - CatBoost: iterations, depth, learning rate, L2 regularization
  - LSTM: layers, units, dropout, batch size, sequence length
  - Both: PCA components (10-200)

**MLflow Integration**:
- Nested runs: Parent study + child trials
- Automatic logging: parameters, metrics, artifacts
- Model versioning and comparison via UI

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Core ML** | CatBoost, TensorFlow/Keras, scikit-learn |
| **Optimization** | Optuna (TPE sampler, MedianPruner) |
| **Experiment Tracking** | MLflow (SQLite backend) |
| **Data Processing** | pandas, NumPy, scipy |
| **Visualization** | Matplotlib, Plotly |
| **Reproducibility** | Docker, YAML configs, joblib serialization |
| **Storage** | SQLite (MLflow + Optuna), Parquet (data) |

---

## 📈 Model Details

### CatBoost Classifier/Regressor

**Architecture**: Gradient boosting on decision trees with ordered boosting  
**Advantages**:
- Handles mixed data types
- Built-in categorical feature support
- Fast training (<2 min per fold)

**Best Parameters** (Classification):
```yaml
iterations: 1300
depth: 6
learning_rate: 0.0094
l2_leaf_reg: 13.2
n_components: 153 (PCA)
```

### LSTM Multi-Output Regressor

**Architecture**: 2-layer LSTM → Dense output (3 gas concentrations)  
**Sequence Handling**: Sliding windows (look_back=58) within each gas type  
**Advantages**:
- Captures temporal dependencies
- Handles variable-length sequences per gas

**Best Parameters**:
```yaml
look_back: 58 timesteps
n_components: 140 (PCA)
n_layers: 2
n_units: 80 (first layer), 40 (second layer)
dropout: 0.13
learning_rate: 0.0028
batch_size: 64
```

---

## 📊 Experiment Tracking

All experiments are logged to **MLflow** with SQLite backend:

```bash
# View all experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access specific experiment
mlflow experiments list --tracking-uri sqlite:///mlflow.db
```

**What's Logged**:
- Hyperparameters (learning rate, depth, dropout, etc.)
- Metrics per fold (RMSE, F1-score, accuracy)
- Aggregated metrics (mean, std, CV stability)
- Model artifacts (CatBoost .cbm, Keras .h5)
- Training curves (loss, validation loss)

**Optuna Study Database**:
```python
import optuna
study = optuna.load_study(
    study_name='lstm_regression_v2',
    storage='sqlite:///optuna_studies.db'
)
print(f"Best RMSE: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

---

## 🧪 Validation Strategy

### Time-Series Cross-Validation

- **Folds**: 8 (expanding window)
- **Train size**: Increases from 6 → 10 cycles
- **Test size**: 1 cycle per fold
- **Shuffle**: False (preserves temporal order)

### Metrics

**Classification**:
- F1-score (macro): Balanced metric for multi-class
- Accuracy: Overall correctness
- Per-class F1: Identifies gas-specific performance

**Regression**:
- RMSE: Root mean squared error (ppm)
- MAE: Mean absolute error (ppm)
- Per-output RMSE: Individual gas predictions

---

## 🔧 Configuration System

All experiments are config-driven for reproducibility:

```yaml
# configs/config_lstm_regression.yaml
experiment:
  name: "SWCNT_LSTM_Regression_v2"
  study_name: "lstm_regression_v2"

data:
  dedrift_method: "Exp_pd"
  dedrift_params:
    alpha: 0.0217
    envelope_ind: [201]

preprocessing:
  cv_start_cycle: 6
  cv_test_size: 2
  look_back_range: [20, 100]
  n_components_range: [10, 100]

optimization:
  n_trials: 100
  direction: "minimize"
  pruner:
    n_startup_trials: 5
    n_warmup_steps: 1
```

**Benefits**:
- Reproducible experiments (version control configs)
- Easy A/B testing (modify YAML, rerun)
- Clear documentation of experiment setup

---

## 📚 Documentation

- **[Methodology](docs/methodology.md)**: Detailed technical approach
- **[Docker Setup](docs/DOCKER_SETUP.md)**: Container deployment guide
- **[Data README](data/README.md)**: Raw data description
- **Notebooks**: Interactive analysis in `notebooks/`

---

## 🎓 Skills Demonstrated

This project showcases:

✅ **End-to-end ML pipeline development** (data → model → evaluation)  
✅ **Time-series analysis** (proper CV, no leakage)  
✅ **Hyperparameter optimization** (Optuna with 100+ trials)  
✅ **Experiment tracking** (MLflow integration)  
✅ **Production engineering** (Docker, config-driven, modular code)  
✅ **Model comparison** (CatBoost vs LSTM trade-offs)  
✅ **Feature engineering** (dedrifting, PCA)  
✅ **Scientific communication** (published research, clear docs)

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

GPL-3.0 License - see [LICENSE](LICENSE) for details

---

## 👤 Author

**Konstantin Zamansky**  
📧 [Contact via LinkedIn](https://www.linkedin.com/in/konstantin-zamansky-244837354/)  
🔬 [ORCID](https://orcid.org/0009-0005-6495-1985)

---

## 🙏 Acknowledgments

- Research published in *Sensors and Actuators B: Chemical*
- CatBoost team for excellent gradient boosting library
- Optuna developers for flexible hyperparameter optimization
- MLflow community for experiment tracking tools

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{zamansky_gas_2024,
	title = {A gas sensor based on free-standing {SWCNT} film for selective recognition of toxic and flammable gases under thermal cycling protocols},
	volume = {417},
	copyright = {All rights reserved},
	issn = {0925-4005},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S0925400524008463},
	doi = {10.1016/j.snb.2024.136116},
	abstract = {The widespread adoption of e-nose devices based on chemiresistive materials has been hindered by issues related to sensor device complexity and reliability, specifically sensor drift, necessitating frequent recalibration and retraining of pattern recognition models. This study introduces a method for thermocycling a single sensor based on a free-standing network of single-walled carbon nanotubes (SWCNTs) to acquire signal patterns for selective analyte detection. Additionally, it employs a data filtering technique to compensate for the sensor drift. A free-standing SWCNT film, only a few nanometers thick, is thermally cycled via Joule heating between room temperature and 120 °C. Under these conditions, the sensitivity was tested towards NO2, H2S, and acetone vapors (10–25 ppm) in the mixture with dry air. Signal patterns produced through thermocycling were processed using CatBoost and LSTM algorithms. The accuracy of detection reached 90 \% in the classification task, and the average root mean squared error of analyte concentration detection in the multioutput regression task was below 4 ppm. By combining original sensor design, thermocycling, signal filtering for drift compensation, and advanced pattern recognition models, this work contributes to overcoming the challenges in multivariate sensing systems, paving the way for practical applications of the more reliable chemiresistive sensors.},
	urldate = {2024-07-04},
	journal = {Sensors and Actuators B: Chemical},
	author = {Zamansky, Konstantin K. and Fedorov, Fedor S. and Shandakov, Sergey D. and Chetyrkina, Margarita and Nasibulin, Albert G.},
	month = oct,
	year = {2024},
	keywords = {Gas sensor, Single-walled carbon nanotubes, Pattern recognition, Мои публикации, Drift compensation, Thermocycling,},
	pages = {136116},
}
```
