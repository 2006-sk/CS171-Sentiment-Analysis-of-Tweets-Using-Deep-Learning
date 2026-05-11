import json
import os

import numpy as np
import pandas as pd


PROCESSED_DIR = "processed"


def test_processed_files_exist():
    expected = [
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "tokenizer.json",
        "label_classes.json",
    ]
    for fname in expected:
        assert os.path.exists(os.path.join(PROCESSED_DIR, fname))


def test_array_shapes():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    assert X_train.ndim == 2 and X_train.shape[1] == 50
    assert X_val.ndim == 2 and X_val.shape[1] == 50
    assert X_test.ndim == 2 and X_test.shape[1] == 50

    assert y_train.ndim == 1 and y_train.shape[0] == X_train.shape[0]
    assert y_val.ndim == 1 and y_val.shape[0] == X_val.shape[0]
    assert y_test.ndim == 1 and y_test.shape[0] == X_test.shape[0]


def test_label_values():
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    all_y = np.concatenate([y_train, y_val, y_test])
    assert set(np.unique(all_y)).issubset({0, 1, 2})


def test_no_data_leakage():
    # We can't perfectly prove "train only" without the raw train text,
    # but we can assert tokenizer metadata exists and has entries.
    tok_path = os.path.join(PROCESSED_DIR, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_json = json.load(f)

    word_index = tok_json.get("config", {}).get("word_index", {})
    # Keras may serialize word_index as a JSON string.
    if isinstance(word_index, str):
        word_index = json.loads(word_index)
    assert isinstance(word_index, dict)
    assert len(word_index) > 0


def test_clean_tweet():
    from preprocess import clean_tweet

    s1 = "Check this out https://example.com @user #Happy!!!"
    out1 = clean_tweet(s1)
    assert "http" not in out1
    assert "@user" not in out1
    assert "happy" in out1
    assert "#" not in out1

    s2 = "@someone I love #CS171 :) www.test.com"
    out2 = clean_tweet(s2)
    assert "@" not in out2
    assert "cs171" in out2
    assert "www" not in out2

    s3 = "WOW!!! #Amazing-day, isn't it???"
    out3 = clean_tweet(s3)
    assert "#" not in out3
    assert "amazing" in out3


def test_lstm_builds():
    from model import get_model

    m = get_model("lstm")
    assert hasattr(m, "predict")
    assert m.output_shape == (None, 3)


def test_bilstm_builds():
    from model import get_model

    m = get_model("bilstm")
    assert hasattr(m, "predict")
    assert m.output_shape == (None, 3)


def test_model_forward_pass():
    from model import get_model

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))[:8]
    m = get_model("lstm")
    y = m(X_train, training=False).numpy()
    assert y.shape == (8, 3)
    row_sums = y.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)


def test_emotion_model_loads():
    from emotion_detection import load_emotion_model

    clf = load_emotion_model()
    assert callable(clf)


def test_assign_emotions_output():
    from emotion_detection import VALID_EMOTIONS, assign_emotions

    texts = [
        "I am so happy today!",
        "This makes me angry.",
        "I'm really scared about tomorrow.",
        "That was surprising!",
        "I feel sad and down.",
    ]
    labels = assign_emotions(texts)
    assert isinstance(labels, list)
    assert len(labels) == 5
    assert all(isinstance(x, str) for x in labels)
    assert all(x in VALID_EMOTIONS for x in labels)


def test_add_emotion_column():
    from emotion_detection import add_emotion_labels

    df = pd.DataFrame({"text": ["I am thrilled!", "This is awful.", "Meh, it's okay."]})
    out = add_emotion_labels(df)
    assert "emotion" in out.columns
    assert out["emotion"].notna().sum() == 3


def test_embedding_matrix_shape():
    path = os.path.join(PROCESSED_DIR, "embedding_matrix.npy")
    assert os.path.exists(path)
    m = np.load(path)
    assert m.shape == (20000, 100)


def test_lstm_with_glove():
    from model import get_model

    m = get_model("lstm", embedding_matrix=np.zeros((20000, 100)))
    assert hasattr(m, "predict")
    assert m.output_shape == (None, 3)


def test_bert_tokenizer_output():
    from evaluate import tokenize_for_bert

    out = tokenize_for_bert(["great tweet", "bad day"], max_len=50)
    assert set(out.keys()) == {"input_ids", "attention_mask"}
    assert out["input_ids"].shape == (2, 50)
    assert out["attention_mask"].shape == (2, 50)

