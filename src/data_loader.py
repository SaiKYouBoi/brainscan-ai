import os, random
import cv2, numpy as np
from .config import valid_ext, image_size, data_dir


def clean_dataset(data_dir):
    removed = 0
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if os.path.splitext(f)[1].lower() not in valid_ext:
                os.remove(os.path.join(cls_dir, f))
                removed += 1
    print(f"Removed {removed} invalid files")


def load_dataset(data_dir, img_size=image_size):
    classes = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    print("Classes:", classes)
    images, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for f in os.listdir(cls_dir):
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


X, y, classes = load_dataset(data_dir)
print(X.shape, y.shape)
