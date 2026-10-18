import os
import random

import cv2
import numpy as np

from .config import DATA_DIR, IMG_SIZE, VALID_EXT


def clean_dataset(root):
    removed = 0
    for cls in os.listdir(root):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            path = os.path.join(cls_dir, f)
            if os.path.isfile(path) and os.path.splitext(f)[1].lower() not in VALID_EXT:
                os.remove(path)
                removed += 1
    print(f"Removed {removed} invalid files")


def load_dataset(root, img_size=IMG_SIZE):
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    print("Classes:", classes)
    images, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        for f in sorted(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, f)
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("unreadable or corrupted image")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(cls)
            except Exception as e:
                print(f"Skipped {path}: {e}")

    combined = list(zip(images, labels))
    if not combined:
        raise ValueError("No valid images found in dataset")
    random.shuffle(combined)
    images, labels = zip(*combined)
    return np.array(images, dtype=np.uint8), np.array(labels), classes


if __name__ == "__main__":
    X, y, classes = load_dataset(DATA_DIR)
    print(X.shape, y.shape)
