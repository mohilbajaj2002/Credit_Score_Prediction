import pathlib
import pandas as pd
import regression_model

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
ALL_VARS_FILE = DATASET_DIR / 'feature_list.txt'
FINAL_VARS_FILE = DATASET_DIR / 'diff3.txt'
CATEGORICAL_VARS_FILE = DATASET_DIR / 'cat_vars.txt'
NUMERICAL_VARS_FILE = DATASET_DIR / 'num_vars.txt'
NUMERICAL_LOG_VARS_FILE = DATASET_DIR / 'num_log_vars.txt'

PIPELINE_NAME = "lasso_regression"
PIPELINE_SAVE_FILE = TRAINED_MODEL_DIR / f"{PIPELINE_NAME}_output_v"

TARGET = 'y'
alpha = 0.05
