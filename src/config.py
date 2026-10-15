import random
import numpy as np
import tensorflow as tf
from pathlib import Path

seed, image_size = 42,224
data_dir = "data/raw"
valid_ext = {".jpeg",".jpg",".bmp",".png"}
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "raw"


def set_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
