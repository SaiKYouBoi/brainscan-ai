import json

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from .config import IMG_SIZE, MODEL_DIR


def predict_image(path, img_size=IMG_SIZE):
    model = load_model(MODEL_DIR / "best_model.h5")
    with open(MODEL_DIR / "classes.json") as f:
        classes = json.load(f)

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype("float32") / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    k = int(np.argmax(probs))
    return classes[k], float(probs[k])
