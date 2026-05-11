import os
from typing import Any, Dict, List, Literal, Tuple

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import get_model
from preprocess import (
    clean_tweet_text,
    drop_nulls_and_duplicates,
    encode_sentiments_with_label_encoder,
    load_csv,
    stratified_train_val_test_split,
    tokenize_and_remove_stopwords,
)

RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
EPOCHS_SEQ = 15
EPOCHS_BERT = 3
N_SPLITS = 5
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def tokenize_for_bert(
    texts: List[str] | np.ndarray,
    max_len: int = 50,
) -> Dict[str, np.ndarray]:
    """Tokenize raw strings for DistilBERT; returns numpy input_ids and attention_mask."""
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


def load_raw_texts(split: Literal["train", "val", "test"] = "test") -> np.ndarray:
    """
    Reproduce preprocess splits (80/10/10, random_state=42) on cleaned text strings.
    """
    split_l = split.lower()
    if split_l not in ("train", "val", "test"):
        raise ValueError('split must be "train", "val", or "test"')

    df = load_csv("data/Tweets.csv")
    df = drop_nulls_and_duplicates(df)
    df["clean_text"] = df["text"].map(clean_tweet_text)
    tokens = tokenize_and_remove_stopwords(df["clean_text"].tolist())
    texts_for_keras = np.array([" ".join(toks) for toks in tokens], dtype=object)

    y, _ = encode_sentiments_with_label_encoder(df["sentiment"].tolist())
    X_train_txt, X_val_txt, X_test_txt, _, _, _ = stratified_train_val_test_split(
        texts_for_keras.tolist(),
        y,
        random_state=42,
    )

    if split_l == "train":
        return np.array(X_train_txt, dtype=object)
    if split_l == "val":
        return np.array(X_val_txt, dtype=object)
    return np.array(X_test_txt, dtype=object)


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


def _callbacks(model_name: str, fold: int | str) -> List[Any]:
    fold_tag = str(fold)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=os.path.join(
                CHECKPOINT_DIR,
                f"{model_name}_fold{fold_tag}.keras",
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
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
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=_callbacks(model_name, fold),
    )
    preds_prob = model.predict(X_val, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return acc, f1, preds, history


def evaluate_bert_model(
    model,
    train_batch: Dict[str, np.ndarray],
    Y_train: np.ndarray,
    val_batch: Dict[str, np.ndarray],
    y_val: np.ndarray,
    epochs: int,
    model_name: str,
    fold: int,
) -> Tuple[float, float, np.ndarray, Any]:
    history = model.fit(
        train_batch,
        Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(val_batch, y_val),
        callbacks=_callbacks(model_name, fold),
    )
    preds_prob = model.predict(val_batch, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return acc, f1, preds, history


def cross_validate_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    embedding_matrix: np.ndarray,
    n_splits: int = N_SPLITS,
) -> List[Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining fold {fold} for {model_name}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_model(model_name, embedding_matrix=embedding_matrix)

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


def cross_validate_bert_model(
    train_texts: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
) -> List[Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: List[Dict[str, Any]] = []
    model_name = "bert"

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y), 1):
        print(f"\nTraining fold {fold} for {model_name}...")
        X_train_txt = train_texts[train_idx].tolist()
        X_val_txt = train_texts[val_idx].tolist()
        y_train, y_val = y[train_idx], y[val_idx]

        train_batch = tokenize_for_bert(X_train_txt)
        val_batch = tokenize_for_bert(X_val_txt)

        model = get_model("bert")
        acc, f1, preds, history = evaluate_bert_model(
            model,
            train_batch,
            y_train,
            val_batch,
            y_val,
            EPOCHS_BERT,
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
    embedding_matrix: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    print(f"\nRunning FINAL TEST evaluation for {model_name}...")
    model = get_model(model_name, embedding_matrix=embedding_matrix)
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_SEQ,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=os.path.join(CHECKPOINT_DIR, f"{model_name}_final.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ],
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


def final_test_bert(
    train_texts: np.ndarray,
    val_texts: np.ndarray,
    test_texts: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model_name = "bert"
    print(f"\nRunning FINAL TEST evaluation for {model_name}...")
    train_b = tokenize_for_bert(train_texts.tolist())
    val_b = tokenize_for_bert(val_texts.tolist())
    test_b = tokenize_for_bert(test_texts.tolist())

    model = get_model("bert")
    model.fit(
        train_b,
        y_train,
        epochs=EPOCHS_BERT,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(val_b, y_val),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=os.path.join(CHECKPOINT_DIR, f"{model_name}_final.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ],
    )
    preds_prob = model.predict(test_b, verbose=0)
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
    print("Loading processed arrays and embedding matrix...")
    X_train = np.load("processed/X_train.npy")
    y_train = np.load("processed/y_train.npy")
    X_val = np.load("processed/X_val.npy")
    y_val = np.load("processed/y_val.npy")
    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")
    embedding_matrix = np.load("processed/embedding_matrix.npy")

    train_texts = load_raw_texts("train")
    val_texts = load_raw_texts("val")
    test_texts = load_raw_texts("test")

    if len(train_texts) != len(y_train):
        raise RuntimeError("train texts length mismatch with y_train")
    if len(val_texts) != len(y_val):
        raise RuntimeError("val texts length mismatch with y_val")
    if len(test_texts) != len(y_test):
        raise RuntimeError("test texts length mismatch with y_test")

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
                embedding_matrix,
                n_splits=N_SPLITS,
            )
        )

    print(f"\n==============================")
    print("Evaluating bert (cross-validation)")
    print(f"==============================")
    all_results.extend(cross_validate_bert_model(train_texts, y_train, n_splits=N_SPLITS))

    save_cv_results_csv(all_results)
    save_summary_csv(all_results)

    final_rows = [
        final_test_sequence(
            "lstm",
            embedding_matrix,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ),
        final_test_sequence(
            "bilstm",
            embedding_matrix,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ),
        final_test_bert(
            train_texts,
            val_texts,
            test_texts,
            y_train,
            y_val,
            y_test,
        ),
    ]
    save_final_results_csv(final_rows)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
