import json
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    Clean tweet text:
    - remove URLs
    - remove @mentions
    - strip '#' from hashtags (keep the word)
    - remove punctuation/special characters
    - lowercase
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Strip '#' but keep hashtag content
    text = text.replace("#", "")

    # Remove punctuation and special characters (keep letters/numbers/whitespace)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tweet(text: str) -> str:
    """
    Backwards-compatible alias used by the test suite.
    """
    return clean_tweet_text(text)


def _ensure_nltk_resources() -> None:
    """
    Ensure NLTK tokenizers and stopwords are available.
    """
    import nltk

    # punkt is needed for word_tokenize; newer NLTK sometimes needs punkt_tab too.
    for resource in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.data.find(
                "tokenizers/punkt"
                if resource.startswith("punkt")
                else f"corpora/{resource}"
            )
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # Some environments (e.g., locked-down SSL certs) may block downloads.
                # We'll fall back to tokenizers/stopwords that don't require downloads.
                return


def tokenize_and_remove_stopwords(texts: List[str]) -> List[List[str]]:
    """
    Tokenize with NLTK word_tokenize and remove English stopwords.
    """
    _ensure_nltk_resources()

    # Prefer NLTK's stopwords if available; otherwise fall back to sklearn's list.
    try:
        from nltk.corpus import stopwords  # type: ignore

        stop_words = set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words = set(ENGLISH_STOP_WORDS)

    # Prefer word_tokenize (requires punkt). If runtime resources are missing (common
    # in environments that can't download NLTK data), fall back to TweetTokenizer.
    from nltk.tokenize import TweetTokenizer  # type: ignore

    tt = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    try:
        from nltk.tokenize import word_tokenize  # type: ignore

        def tokenize_fn(s: str) -> List[str]:
            try:
                return word_tokenize(s)
            except LookupError:
                return tt.tokenize(s)

    except Exception:

        def tokenize_fn(s: str) -> List[str]:
            return tt.tokenize(s)

    tokenized: List[List[str]] = []
    for t in texts:
        tokens = tokenize_fn(t)
        tokens = [tok for tok in tokens if tok not in stop_words and tok.strip()]
        tokenized.append(tokens)
    return tokenized


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


def fit_tokenizer_on_train_and_pad(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    vocab_size: int = 20000,
    oov_token: str = "<OOV>",
    maxlen: int = 50,
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


def save_raw_text_for_bert(
    out_dir: str,
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Save clean raw text files for BERT (no tokenization, no padding).
    This is required for HuggingFace tokenizers later.
    """
    os.makedirs(out_dir, exist_ok=True)

    def save_csv(path: str, texts: List[str], labels: np.ndarray) -> None:
        df = pd.DataFrame({"label": labels, "text": texts})
        df.to_csv(path, index=False, encoding="utf-8")

    save_csv(
        os.path.join(out_dir, "bert_train.csv"),
        train_texts,
        y_train,
    )

    save_csv(
        os.path.join(out_dir, "bert_val.csv"),
        val_texts,
        y_val,
    )

    save_csv(
        os.path.join(out_dir, "bert_test.csv"),
        test_texts,
        y_test,
    )


def run_preprocessing(csv_path: str) -> Dict[str, Any]:
    """
    Chain all preprocessing steps and return a dict with splits + tokenizer + label encoder.
    Also saves outputs into processed/.
    """
    df = load_csv(csv_path)
    df = drop_nulls_and_duplicates(df)

    df["clean_text"] = df["text"].map(clean_tweet_text)
    raw_texts = df["text"].map(clean_tweet_text).tolist()

    # Tokenize + stopword removal (NLTK)
    tokens = tokenize_and_remove_stopwords(df["clean_text"].tolist())
    texts_for_keras = [" ".join(toks) for toks in tokens]

    # Encode sentiments
    y, label_encoder = encode_sentiments_with_label_encoder(df["sentiment"].tolist())

    # Stratified split
    X_train_txt, X_val_txt, X_test_txt, y_train, y_val, y_test = (
        stratified_train_val_test_split(texts_for_keras, y)
    )

    # Tokenizer + padding
    split_data_no_y, tokenizer = fit_tokenizer_on_train_and_pad(
        X_train_txt,
        X_val_txt,
        X_test_txt,
        vocab_size=20000,
        oov_token="<OOV>",
        maxlen=50,
    )

    X_train_txt, X_val_txt, X_test_txt, y_train, y_val, y_test = (
        stratified_train_val_test_split(raw_texts, y)
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

    save_raw_text_for_bert(
        out_dir="processed",
        train_texts=X_train_txt,
        val_texts=X_val_txt,
        test_texts=X_test_txt,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
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
    print(f"Label classes (index order): {outputs['label_encoder'].classes_.tolist()}")
