"""
test_setup.py

Quick test script to verify all components work correctly.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path.cwd().parents[0]
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        from src.data.loading import load_gas_data
        from src.data.cleaning import apply_manual_trim
        from src.data.assemble import build_basic_dataset, full_dataset
        from src.preprocessing.smoothing import Exp_pd, Savitzky_Golay, dedrift
        from src.preprocessing.train_test import create_time_series_folds
        from src.models.catboost_model import build_catboost_classifier, build_catboost_regressor
        from src.models.lstm_model import build_lstm
        from src.models.optuna_objectives import (
            LSTMRegressorObjective,
            CatBoostClassifierObjective
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test data loading pipeline."""
    print("\nTesting data loading...")
    
    try:
        from src.data.assemble import full_dataset
        from src.preprocessing.smoothing import Exp_pd
        
        df = full_dataset(
            dedrifting_func=Exp_pd,
            envelope_ind=[201],
            alpha=0.0217
        )
        
        assert df.shape[1] == 408, f"Expected 408 columns, got {df.shape[1]}"
        assert 'NO2' in df.columns, "Missing NO2 column"
        assert 'class_' in df.columns, "Missing class_ column"
        assert 'meas_cycle' in df.columns, "Missing meas_cycle column"
        
        print(f"✓ Data loaded successfully: {df.shape}")
        print(f"  Gases: {df['gas_'].unique()}")
        print(f"  Cycles: {df['meas_cycle'].min()} - {df['meas_cycle'].max()}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_cv_splitting():
    """Test CV splitting for both model types."""
    print("\nTesting CV splitting...")
    
    try:
        from src.data.assemble import full_dataset
        from src.preprocessing.smoothing import Exp_pd
        from src.preprocessing.train_test import create_time_series_folds
        
        df = full_dataset(
            dedrifting_func=Exp_pd,
            envelope_ind=[201],
            alpha=0.0217
        )
        
        # Test CatBoost folds
        catboost_folds = create_time_series_folds(
            df,
            model_type='catboost',
            task_type='regressor',
            n_components=50,
            start_cycle=7,
            test_size=1
        )
        print(f"✓ CatBoost folds: {len(catboost_folds)} folds")
        
        # Test LSTM folds
        lstm_folds = create_time_series_folds(
            df,
            model_type='lstm',
            task_type='regressor',
            look_back=30,
            n_components=50,
            start_cycle=7,
            test_size=1
        )
        print(f"✓ LSTM folds: {len(lstm_folds)} folds")
        
        return True
    except Exception as e:
        print(f"✗ CV splitting failed: {e}")
        return False


def test_model_building():
    """Test model building."""
    print("\nTesting model building...")
    
    try:
        from src.models.catboost_model import build_catboost_regressor
        from src.models.lstm_model import build_lstm
        
        # Test CatBoost
        catboost_model = build_catboost_regressor(iterations=10, verbose=False)
        print("✓ CatBoost model built")
        
        # Test LSTM
        lstm_model = build_lstm(
            input_shape=(30, 50),  # (look_back, n_features)
            output_shape=3,  # 3 gas concentrations
            n_layers=2,
            n_units=32
        )
        print("✓ LSTM model built")
        print(f"  LSTM parameters: {lstm_model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False


def test_config_loading():
    """Test config file loading."""
    print("\nTesting config loading...")
    
    try:
        import yaml
        
        config_files = [
            'configs/lstm_regression.yaml',
            'configs/catboost_classification.yaml',
            'configs/catboost_regression.yaml'
        ]
        
        for config_file in config_files:
            config_path = PROJECT_ROOT / config_file
            if not config_path.exists():
                print(f"⚠ Config file not found: {config_file}")
                continue
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'experiment' in config, f"Missing 'experiment' in {config_file}"
            assert 'data' in config, f"Missing 'data' in {config_file}"
            print(f"✓ Config loaded: {config_file}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def main():
    print("="*60)
    print("TESTING SETUP")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("CV Splitting", test_cv_splitting),
        ("Model Building", test_model_building),
        ("Config Loading", test_config_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
