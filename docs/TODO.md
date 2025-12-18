Updated on 18.12.2025

0. ensure that the main readme is up to date with the latest commits (check examples in the ###Basic usage, check MLFlow UI usage example, access to Optuna trials etc.)

1. ensure config correctness:

  a. remove duplicating look_back and n_components from the model section (leave in preprocessing). Currently the optimization is working, but duplicating params is redundant

  b. adjust parameter propagation in optuna_objectives.py, run_optimization_from_config.py and config.py
  
  *currently it is propagated the following way*
  ```python
  '''run_optimization_from_config.py'''
  cfg = RunConfig.model_validate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
  objective_kwargs = {"model_config": cfg.model,"cv_start_cycle": cfg.preprocessing.cv_start_cycle,...}
  optimize_func = ... run_lstm_regressor_optimization
  study = optimize_func(objective_kwargs=objective_kwargs,...)

  '''optuna_objectives.py'''
  class BaseObjective:
    def __init__(self,df,model_config: Dict[str, Any],):
      self.config = model_config

  class LSTMRegressorObjective(BaseObjective):
    def __call__(self, trial: optuna.Trial):
      params = {"look_back": trial.suggest_int("look_back", *_get_range(self.config, "look_back_range", [20, 60])),...}

  def run_optimization(objective_class, df, objective_kwargs):
    objective = objective_class(df=df, **(objective_kwargs or {}))
    study.optimize(objective,)

  def run_lstm_regressor_optimization(df, **kwargs):
    run_optimization(LSTMRegressorObjective, df, **kwargs)
  
  '''config.py'''
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
  ```
  
  c. ensure, that catboost piplines are not affected and exemplar configs are updated

2. Change the output file logging from ```best metric value and param values to .yaml, used config to .json, optuna trials to .tsv``` to ```best metric value + raw class values, params search ranges + best values + [Optional] some info about training (like how close train to validation was, was validation improvement plateau reached)```

3. Change the default MLflow behaviour from dumping to sql database to UI (introduce changes to optuna_objectives.py, and CLI command examples)

4. Introduce optuna plots saving (plot hyperparameter importances by weight, training time + plot_slices) into one .png image + text on the image explaining config details that are not clear from the graph

5. check the Docker + create DOCKERFILE_README.md

6. fill methodology.md and results.md