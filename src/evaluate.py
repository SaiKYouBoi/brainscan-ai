import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .config import FIGURE_DIR


def evaluate_model(model, X_test, y_test, history, classes):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    # Learning curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_title(metric)
        ax.legend()
    fig.savefig(FIGURE_DIR / "learning_curves.png")
    plt.close(fig)

    # Confusion matrix and per-class metrics.
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png")
    plt.close()
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print(report)
    with open(FIGURE_DIR.parent / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.3f}\nTest loss: {loss:.3f}\n\n{report}")

    # Correct and incorrect examples.
    wrong = np.where(y_pred != y_true)[0][:6]
    right = np.where(y_pred == y_true)[0][:6]
    for title, indices in [("Correct", right), ("Incorrect", wrong)]:
        plt.figure(figsize=(14, 3))
        for i, k in enumerate(indices):
            plt.subplot(1, 6, i + 1)
            plt.imshow(X_test[k])
            plt.axis("off")
            plt.title(f"T:{classes[y_true[k]]}\nP:{classes[y_pred[k]]}", fontsize=8)
        plt.suptitle(title)
        plt.savefig(FIGURE_DIR / f"{title.lower()}_examples.png")
        plt.close()
