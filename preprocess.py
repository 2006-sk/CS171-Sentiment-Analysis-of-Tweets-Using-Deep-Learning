import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_LEN = 30

# -----------------------------
# Step 1: Data Preprocessing
# -----------------------------


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the Kaggle Twitter Tweets Sentiment Dataset CSV.
    Expected columns: textID, text, selected_text, sentiment
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'. Expected: data/Tweets.csv"
        )
    return pd.read_csv(csv_path)


def drop_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with nulls in required columns and duplicate tweets by text.
    """
    required = ["text", "sentiment"]
    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def clean_tweet_text(text: str) -> str:
    """
    Clean tweet text: HTML unescape, URLs/mentions/hashtags, expand contractions,
    then strip non-alphanumeric (after contractions so apostrophes are handled).
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    text = text.lower()
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "it's": "it is",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "you're": "you are",
        "you've": "you have",
        "you'll": "you will",
        "you'd": "you would",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "they've": "they have",
        "they'll": "they will",
        "we're": "we are",
        "we've": "we have",
        "we'll": "we will",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "didn't": "did not",
        "doesn't": "does not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tweet(text: str) -> str:
    """
    Backwards-compatible alias used by the test suite.
    """
    return clean_tweet_text(text)


def tokenize_text(text: str) -> str:
    """
    Regex tokenization: lowercase words of 2+ letters only (no NLTK, no downloads).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = re.findall(r"\b[a-z][a-z]+\b", text)
    return " ".join(tokens)


def encode_sentiments_with_label_encoder(
    sentiments: List[str],
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode sentiment labels with LabelEncoder using fixed class order:
    negative=0, neutral=1, positive=2
    """
    le = LabelEncoder()
    le.fit(["negative", "neutral", "positive"])
    y = le.transform(sentiments)
    return y.astype(np.int64), le


def stratified_train_val_test_split(
    X: List[str],
    y: np.ndarray,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified 80/10/10 train/val/test split using train_test_split.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    # Split remaining equally into val/test
    rel_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=rel_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_train_val_test_indices(
    y: np.ndarray,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same stratification as stratified_train_val_test_split, but return row indices
    so Keras tokenized lines and raw BERT texts stay aligned split-wise.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    idx = np.arange(len(y), dtype=np.int64)
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )
    rel_test_size = test_size / (val_size + test_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=rel_test_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return idx_train, idx_val, idx_test


def save_bert_text_csvs(
    out_dir: str,
    texts_bert: List[str],
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> None:
    """Train+val as bert_train.csv, held-out test as bert_test.csv (text, label ints)."""
    os.makedirs(out_dir, exist_ok=True)
    tr = [texts_bert[i] for i in idx_train]
    va = [texts_bert[i] for i in idx_val]
    te = [texts_bert[i] for i in idx_test]
    y_tr = y[idx_train]
    y_va = y[idx_val]
    y_te = y[idx_test]
    train_df = pd.DataFrame(
        {"text": tr + va, "label": np.concatenate([y_tr, y_va]).astype(np.int64)}
    )
    test_df = pd.DataFrame({"text": te, "label": y_te.astype(np.int64)})
    train_df.to_csv(os.path.join(out_dir, "bert_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "bert_test.csv"), index=False)


def fit_tokenizer_on_train_and_pad(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    vocab_size: int = 20000,
    oov_token: str = "<OOV>",
    maxlen: int = MAX_LEN,
) -> Tuple[SplitData, Tokenizer]:
    """
    Fit a Keras Tokenizer on train only, then pad all splits to maxlen.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    val_seq = tokenizer.texts_to_sequences(val_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)

    X_train = pad_sequences(train_seq, maxlen=maxlen, padding="post", truncating="post")
    X_val = pad_sequences(val_seq, maxlen=maxlen, padding="post", truncating="post")
    X_test = pad_sequences(test_seq, maxlen=maxlen, padding="post", truncating="post")

    return (
        SplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=np.array([]),
            y_val=np.array([]),
            y_test=np.array([]),
        ),
        tokenizer,
    )


def save_outputs(
    out_dir: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tokenizer: Tokenizer,
    label_encoder: LabelEncoder,
) -> None:
    """
    Save processed arrays and metadata to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    tokenizer_json = tokenizer.to_json()
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer_json)

    label_classes = [str(c) for c in label_encoder.classes_.tolist()]
    with open(os.path.join(out_dir, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(label_classes, f, ensure_ascii=False, indent=2)


def run_preprocessing(csv_path: str) -> Dict[str, Any]:
    """
    Chain all preprocessing steps and return a dict with splits + tokenizer + label encoder.
    Also saves outputs into processed/.
    """
    df = load_csv(csv_path)
    df = drop_nulls_and_duplicates(df)

    df["clean_text"] = df["text"].map(clean_tweet_text)

    texts_for_keras = [tokenize_text(t) for t in df["clean_text"].tolist()]
    texts_bert = df["clean_text"].tolist()

    # Encode sentiments
    y, label_encoder = encode_sentiments_with_label_encoder(df["sentiment"].tolist())

    idx_train, idx_val, idx_test = stratified_train_val_test_indices(y)
    X_train_txt = [texts_for_keras[i] for i in idx_train]
    X_val_txt = [texts_for_keras[i] for i in idx_val]
    X_test_txt = [texts_for_keras[i] for i in idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    # Tokenizer + padding
    split_data_no_y, tokenizer = fit_tokenizer_on_train_and_pad(
        X_train_txt,
        X_val_txt,
        X_test_txt,
        vocab_size=20000,
        oov_token="<OOV>",
        maxlen=MAX_LEN,
    )

    split_data = SplitData(
        X_train=split_data_no_y.X_train,
        X_val=split_data_no_y.X_val,
        X_test=split_data_no_y.X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # Save
    save_outputs(
        out_dir="processed",
        X_train=split_data.X_train,
        X_val=split_data.X_val,
        X_test=split_data.X_test,
        y_train=split_data.y_train,
        y_val=split_data.y_val,
        y_test=split_data.y_test,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
    )
    save_bert_text_csvs(
        "processed",
        texts_bert,
        y,
        idx_train,
        idx_val,
        idx_test,
    )

    return {
        "X_train": split_data.X_train,
        "X_val": split_data.X_val,
        "X_test": split_data.X_test,
        "y_train": split_data.y_train,
        "y_val": split_data.y_val,
        "y_test": split_data.y_test,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
    }


if __name__ == "__main__":
    outputs = run_preprocessing("data/Tweets.csv")
    print("Preprocessing complete.")
    print(
        f"Shapes: X_train={outputs['X_train'].shape}, X_val={outputs['X_val'].shape}, X_test={outputs['X_test'].shape}"
    )
    print(
        f"Label classes (index order): {outputs['label_encoder'].classes_.tolist()}"
    )
