import pathlib
import pandas as pd

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
FEATURE_FILE = DATASET_DIR / 'feature_list.txt'
TARGET = 'y'

f = open(FEATURE_FILE, "r")
FEATURES = f.read()


def save_pipeline() -> None:
    """Persist the pipeline."""

    pass


def run_training() -> None:
    """Train the model."""

    print('Training...')


if __name__ == '__main__':
    run_training()
