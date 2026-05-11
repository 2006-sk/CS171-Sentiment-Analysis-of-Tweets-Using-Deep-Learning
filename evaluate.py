import os
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from model import get_model

RESULTS_DIR = "results"
MAX_LEN = 30
EPOCHS_SEQ = 10
N_SPLITS = 3
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 3

os.makedirs(RESULTS_DIR, exist_ok=True)


def sequence_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(enumerate(weights))


def tokenize_for_bert(
    texts: List[str] | np.ndarray,
    max_len: int = MAX_LEN,
) -> Dict[str, np.ndarray]:
    """Kept for tests; not used in main evaluation."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tok(
        list(texts),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return {
        "input_ids": np.asarray(enc["input_ids"], dtype=np.int32),
        "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int32),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, label) -> None:
    cm = confusion_matrix(y_true, y_pred)
    display_labels = ["Negative", "Neutral", "Positive"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.title(f"{model_name} - {label}")
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/{model_name}_{label}_cm.png"
    plt.savefig(filename)
    plt.close()


def plot_accuracy_loss_curve(history, model_name, fold) -> None:
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))

    ax_acc.plot(history.history["accuracy"], label="Train Accuracy")
    ax_acc.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    ax_loss.plot(history.history["loss"], label="Train Loss")
    ax_loss.plot(history.history["val_loss"], label="Val Loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    plt.suptitle(f"{model_name} - Fold {fold}")
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/{model_name}_fold_{fold}_accuracy_loss_curve.png"
    plt.savefig(filename)
    plt.close()


def _early_stopping_val_acc() -> List[Any]:
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
    ]


def evaluate_sequence_model(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    model_name: str,
    fold: int,
) -> Tuple[float, float, np.ndarray, Any]:
    class_weight_dict = sequence_class_weights(Y_train)
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=_early_stopping_val_acc(),
    )
    preds_prob = model.predict(X_val, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return acc, f1, preds, history


def cross_validate_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = N_SPLITS,
) -> List[Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining fold {fold} for {model_name}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_model(model_name)

        acc, f1, preds, history = evaluate_sequence_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            EPOCHS_SEQ,
            model_name,
            fold,
        )

        results.append(
            {
                "model": model_name,
                "fold": fold,
                "accuracy": acc,
                "f1_score": f1,
            }
        )

        plot_confusion_matrix(y_val, preds, model_name, f"fold_{fold}")
        plot_accuracy_loss_curve(history, model_name, fold)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(f"Fold {fold} F1: {f1:.4f}")
        tf.keras.backend.clear_session()

    return results


def final_test_sequence(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    print(f"\nRunning FINAL TEST evaluation for {model_name}...")
    model = get_model(model_name)
    class_weight_dict = sequence_class_weights(y_train)
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_SEQ,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=_early_stopping_val_acc(),
    )
    preds_prob = model.predict(X_test, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    report = classification_report(
        y_test,
        preds,
        target_names=["Negative", "Neutral", "Positive"],
    )
    plot_confusion_matrix(y_test, preds, model_name, "test")
    print(f"TEST Accuracy: {acc:.4f}")
    print(f"TEST F1: {f1:.4f}")
    tf.keras.backend.clear_session()
    return {
        "model": model_name,
        "test_accuracy": acc,
        "test_f1_weighted": f1,
        "classification_report": report,
    }


def save_cv_results_csv(all_results: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/model_metrics.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}/model_metrics.csv")


def save_summary_csv(all_results: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(all_results)
    summary_rows = []
    for model_name, g in df.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "accuracy_mean": g["accuracy"].mean(),
                "accuracy_std": g["accuracy"].std(),
                "f1_score_mean": g["f1_score"].mean(),
                "f1_score_std": g["f1_score"].std(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        f"{RESULTS_DIR}/model_metrics_summary.csv",
        index=False,
    )
    print(f"Saved summary to {RESULTS_DIR}/model_metrics_summary.csv")


def save_final_results_csv(rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/final_results.csv", index=False)
    print(f"Saved final test results to {RESULTS_DIR}/final_results.csv")


def main() -> None:
    print("Loading processed arrays...")
    X_train = np.load("processed/X_train.npy")
    y_train = np.load("processed/y_train.npy")
    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    all_results: List[Dict[str, Any]] = []

    for model_name in ("lstm", "bilstm"):
        print(f"\n==============================")
        print(f"Evaluating {model_name}")
        print(f"==============================")
        all_results.extend(
            cross_validate_sequence_model(
                X_train,
                y_train,
                model_name,
                n_splits=N_SPLITS,
            )
        )

    save_cv_results_csv(all_results)
    save_summary_csv(all_results)

    final_rows = [
        final_test_sequence(
            "lstm",
            X_train,
            y_train,
            X_test,
            y_test,
        ),
        final_test_sequence(
            "bilstm",
            X_train,
            y_train,
            X_test,
            y_test,
        ),
    ]
    save_final_results_csv(final_rows)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
