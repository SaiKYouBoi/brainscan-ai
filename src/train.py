import json
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model

from .config import DATA_DIR, MODEL_DIR, FIGURE_DIR, SEED, BATCH_SIZE, EPOCHS, set_seed
from .data_loader import load_dataset
from .evaluate import evaluate_model
from .model import build_model


def train():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load and explore the data.
    X, y, classes = load_dataset(DATA_DIR)
    print(X.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=unique, y=counts)
    plt.title("Images per class")
    plt.ylabel("Count")
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.close()

    fig, axes = plt.subplots(len(classes), 5, figsize=(12, 3 * len(classes)), squeeze=False)
    for i, cls in enumerate(classes):
        for ax in axes[i]:
            ax.axis("off")
        for j, k in enumerate(np.where(y == cls)[0][:5]):
            axes[i, j].imshow(X[k])
            if j == 0:
                axes[i, j].set_title(cls, loc="left")
    fig.savefig(FIGURE_DIR / "samples.png")
    plt.close(fig)

    # Encode, split, and balance only the training set.
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    y_cat = to_categorical(y_int)
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump(list(le.classes_), f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_int, random_state=SEED,
    )
    del X
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_int), y=np.argmax(y_train, axis=1),
    )
    class_weights = dict(enumerate(weights))

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    model = build_model(num_classes=len(classes))
    model.summary()
    plot_model(model, to_file=str(FIGURE_DIR / "model.png"), show_shapes=True)

    callbacks = [
        ModelCheckpoint(str(MODEL_DIR / "best_model.h5"), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    start = time.time()
    history = model.fit(
        X_train, y_train, validation_split=0.1, epochs=EPOCHS,
        batch_size=BATCH_SIZE, class_weight=class_weights, callbacks=callbacks,
    )
    print(f"Training time: {(time.time() - start) / 60:.1f} min")

    model = load_model(MODEL_DIR / "best_model.h5")
    evaluate_model(model, X_test, y_test, history, list(le.classes_))


if __name__ == "__main__":
    train()
