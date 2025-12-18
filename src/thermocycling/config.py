"""
config.py

Pydantic models for validating YAML configuration files.
Catches configuration errors before training starts.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, Dict, Any, List, Sequence

Task = Literal["regression", "classification"]
ModelType = Literal["lstm", "catboost"]
Direction = Literal["minimize", "maximize"]
DedriftMethod = Literal["Exp_pd", "Savitzky_Golay"]


class ExperimentConfig(BaseModel):
    """Experiment metadata and task definition."""
    name: str = Field(..., description="MLflow experiment name")
    study_name: str = Field(..., description="Optuna study name")
    task: Task
    model_type: ModelType


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    feature_cols: int = Field(402, gt=0, description="Number of feature columns")
    target_cols: List[str] = Field(..., description="Target column names")
    dedrift_method: DedriftMethod
    dedrift_params: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("target_cols")
    @classmethod
    def validate_targets(cls, v):
        if not v:
            raise ValueError("target_cols must be non-empty")
        return v


class PreprocessingConfig(BaseModel):
    """Cross-validation and feature engineering settings."""
    cv_start_cycle: int = Field(..., ge=0, description="Starting cycle for CV splits")
    cv_test_size: int = Field(..., gt=0, description="Number of cycles per test fold")
    n_components_range: Sequence[int] = Field(..., description="PCA components range [min, max]")
    look_back_range: Optional[Sequence[int]] = Field(None, description="LSTM sequence length range [min, max]")
    
    @field_validator("n_components_range", "look_back_range")
    @classmethod
    def validate_ranges(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"Range must have exactly 2 elements, got {len(v)}")
            if v[0] >= v[1]:
                raise ValueError(f"Range must be [min, max] with min < max, got {v}")
        return v


class PrunerConfig(BaseModel):
    """Optuna pruner configuration."""
    type: str = "MedianPruner"
    n_startup_trials: int = Field(5, ge=0)
    n_warmup_steps: int = Field(1, ge=0)
    interval_steps: int = Field(1, gt=0)


class OptimizationConfig(BaseModel):
    """Optuna optimization settings."""
    n_trials: int = Field(..., gt=0)
    timeout: Optional[int] = Field(None, ge=0, description="Timeout in seconds")
    direction: Direction
    pruner: PrunerConfig = Field(default_factory=PrunerConfig)


class StorageConfig(BaseModel):
    """Storage backend URIs."""
    optuna: str = "sqlite:///optuna_studies.db"
    mlflow: Optional[str] = None  # Deprecated, use mlflow.tracking_uri


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: Optional[str] = None
    log_models: bool = True
    log_plots: bool = True


class RunConfig(BaseModel):
    """Complete run configuration with cross-validation."""
    experiment: ExperimentConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: Dict[str, Any] = Field(..., description="Model-specific hyperparameter ranges")
    optimization: OptimizationConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    random_seed: int = 42
    results_dir: str = "results"
    
    @model_validator(mode="after")
    def validate_cross_dependencies(self):
        """Validate that config sections are consistent with each other."""
        
        # Task-specific target column validation
        task = self.experiment.task
        tcols = self.data.target_cols
        
        if task == "classification":
            if tcols != ["class_"]:
                raise ValueError(
                    f"For classification, target_cols must be ['class_'], got {tcols}"
                )
        else:  # regression
            if "class_" in tcols:
                raise ValueError(
                    f"For regression, target_cols should not contain 'class_', got {tcols}"
                )
            # Common regression targets
            valid_regression_targets = {"NO2", "H2S", "Acet"}
            invalid = set(tcols) - valid_regression_targets
            if invalid:
                raise ValueError(
                    f"Unknown regression targets: {invalid}. "
                    f"Valid: {valid_regression_targets}"
                )
        
        # Model-specific requirements
        if self.experiment.model_type == "lstm":
            if not self.preprocessing.look_back_range:
                raise ValueError(
                    "preprocessing.look_back_range is required for LSTM models"
                )
            
            # Check that model config has required LSTM fields
            required_lstm_fields = [
                "n_layers_range", "n_units_range", "dropout_range",
                "learning_rate_range", "batch_size_options"
            ]
            missing = [f for f in required_lstm_fields if f not in self.model]
            if missing:
                raise ValueError(
                    f"LSTM model config missing required fields: {missing}"
                )
        
        elif self.experiment.model_type == "catboost":
            # Check CatBoost required fields
            required_cb_fields = [
                "iterations_range", "depth_range", 
                "learning_rate_range", "l2_leaf_reg_range"
            ]
            missing = [f for f in required_cb_fields if f not in self.model]
            if missing:
                raise ValueError(
                    f"CatBoost model config missing required fields: {missing}"
                )
        
        return self
    
    @field_validator("model")
    @classmethod
    def validate_model_ranges(cls, v):
        """Ensure all *_range fields are 2-element sequences."""
        for key, val in v.items():
            if key.endswith("_range"):
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(
                        f"Model config field '{key}' must be 2-element [min, max], got {val}"
                    )
                if val[0] >= val[1]:
                    raise ValueError(
                        f"Model config field '{key}' must be [min, max] with min < max, got {val}"
                    )
        return v