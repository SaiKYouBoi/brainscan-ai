import random
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SEED, IMG_SIZE = 42, 224
BATCH_SIZE, EPOCHS = 32, 30
VALID_EXT = {".jpeg", ".jpg", ".bmp", ".png"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Names used by the original EDA notebook.
seed, image_size, data_dir = SEED, IMG_SIZE, DATA_DIR
valid_ext, project_root = VALID_EXT, PROJECT_ROOT
