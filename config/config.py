import os

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, 'data')
MODELS_DIR = Path(BASE_DIR, 'models')
CONFIG_DIR = Path(BASE_DIR, 'config')
CLASSIFIERS_DIR = Path(BASE_DIR, 'classifiers')

ENV_VARIABLES = {
    **os.environ,
}