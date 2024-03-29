import os

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'
CONFIG_DIR = BASE_DIR / 'config'

ENV_VARIABLES = {
    **os.environ,
}