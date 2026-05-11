import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from model import get_model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name, label):
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


def plot_accuracy_loss_curve(history, model_name, fold):
    plt.figure()

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title(f"{model_name} - Fold {fold} Accuracy/Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.tight_layout()

    filename = f"{RESULTS_DIR}/{model_name}_fold_{fold}_accuracy_loss_curve.png"
    plt.savefig(filename)
    plt.close()


def evaluate_model(model, X_train, Y_train, X_val, y_val):
    history = model.fit(
        X_train,
        Y_train,
        epochs=5,
        batch_size=32,
        verbose=0,
        validation_data=(X_val, y_val),
    )

    preds_prob = model.predict(X_val, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")

    return acc, f1, preds, history


def cross_validate_model(X, y, model_name, n_splits=2):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining fold {fold} for {model_name}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_model(model_name)

        acc, f1, preds, history = evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(
            {
                "model": model_name,
                "fold": fold,
                "accuracy": acc,
                "f1_score": f1,
            }
        )

        plot_confusion_matrix(y_val, preds, model_name, fold)
        plot_accuracy_loss_curve(history, model_name, fold)

        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(f"Fold {fold} F1: {f1:.4f}")

        tf.keras.backend.clear_session()

    return results


def final_test_evaluation(model_name, X_train, y_train, X_test, y_test):
    print(f"\nRunning FINAL TEST evaluation for {model_name}...")

    model = get_model(model_name)

    # Train on full training data
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=0,
    )

    preds_prob = model.predict(X_test, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    plot_confusion_matrix(y_test, preds, model_name, "test")

    print(f"TEST Accuracy: {acc:.4f}")
    print(f"TEST F1: {f1:.4f}")

    return {
        "model": model_name,
        "stage": "test",
        "fold": "NA",
        "accuracy": acc,
        "f1_score": f1,
    }


def final_test_evaluation_bert():
    print(f"\nRunning FINAL TEST evaluation for bert...")

    model = get_model("bert")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_texts, train_labels = load_csv("processed/bert_train.csv")

    test_texts, test_labels = load_csv("processed/bert_test.csv")

    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="np",
    )

    test_encodings = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="np",
    )

    X_train = {
        "input_ids": np.array(train_encodings["input_ids"]),
        "attention_mask": np.array(train_encodings["attention_mask"]),
    }

    X_test = {
        "input_ids": np.array(test_encodings["input_ids"]),
        "attention_mask": np.array(test_encodings["attention_mask"]),
    }

    y_train = np.array(train_labels, dtype=np.int32)
    y_test = np.array(test_labels, dtype=np.int32)

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=0,
    )

    preds_prob = model.predict(
        X_test,
        verbose=0,
    )

    y_test = test_labels
    preds = np.argmax(preds_prob, axis=1)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    plot_confusion_matrix(y_test, preds, "bert", "test")

    print(f"TEST Accuracy: {acc:.4f}")
    print(f"TEST F1: {f1:.4f}")

    return {
        "model": "bert",
        "stage": "test",
        "fold": "NA",
        "accuracy": acc,
        "f1_score": f1,
    }


def save_results_csv(all_results):
    df = pd.DataFrame(all_results)

    file_path = f"{RESULTS_DIR}/model_metrics.csv"
    df.to_csv(file_path, index=False)

    # Also save aggregated summary
    summary = df.groupby("model")[["accuracy", "f1_score"]].agg(["mean", "std"])
    cv_summary = df.groupby("model")["f1_score"].mean()
    summary.to_csv(f"{RESULTS_DIR}/model_metrics_summary.csv")

    print(f"\nSaved results to {file_path}")

    best_model = cv_summary.idxmax()
    best_model = str(best_model).strip()
    return best_model


def load_csv(path):
    df = pd.read_csv(path)

    # drop NaNs in BOTH columns
    df = df.dropna(subset=["text", "label"])

    # ensure correct types
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df["text"].tolist(), df["label"].tolist()


def main():
    print("Loading dataset...")

    X = np.load("processed/X_train.npy")
    y = np.load("processed/y_train.npy")

    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    models = ["lstm", "bilstm"]

    all_results = []
    best_model = ""
    for model_name in models:
        print(f"Evaluating {model_name}")

        cv_results = cross_validate_model(X, y, model_name)
        all_results.extend(cv_results)

        best_model = save_results_csv(all_results)

    result = final_test_evaluation_bert()
    df = pd.DataFrame([result])
    df.to_csv(f"{RESULTS_DIR}/bert_final.csv", index=False)

    result = final_test_evaluation(best_model, X, y, X_test, y_test)
    df = pd.DataFrame([result])
    df.to_csv(f"{RESULTS_DIR}/{best_model}_final.csv", index=False)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
