from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
)
from tensorflow.keras.models import Model


def build_lstm_model(
    vocab_size: int = 20000,
    embed_dim: int = 128,
    max_len: int = 30,
    num_classes: int = 3,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Model:
    inputs = tf.keras.Input(shape=(max_len,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=True,
        )(inputs)
    else:
        x = Embedding(vocab_size, embed_dim, trainable=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, output, name="lstm_sentiment")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm_model(
    vocab_size: int = 20000,
    embed_dim: int = 128,
    max_len: int = 30,
    num_classes: int = 3,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Model:
    inputs = tf.keras.Input(shape=(max_len,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=True,
        )(inputs)
    else:
        x = Embedding(vocab_size, embed_dim, trainable=True)(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, output, name="bilstm_sentiment")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bert_model(num_classes: int = 3, max_len: int = 30, **_: Any) -> Any:
    """
    DistilBERT sequence classifier (frozen encoder). Uses tf_keras so Transformers
    accepts symbolic tensors. max_len is accepted for API compatibility with get_model.
    """
    import tf_keras as keras
    from transformers import TFAutoModelForSequenceClassification

    bert = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
        use_safetensors=False,
    )
    if hasattr(bert, "distilbert"):
        bert.distilbert.trainable = False
    else:
        bert.trainable = False

    input_ids = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = keras.layers.Softmax(name="softmax")(logits)
    model = keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=probs,
        name="distilbert_sentiment",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model(
    name: str,
    embedding_matrix: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Any:
    key = name.lower().strip()
    if key == "lstm":
        return build_lstm_model(embedding_matrix=embedding_matrix, **kwargs)
    if key == "bilstm":
        return build_bilstm_model(embedding_matrix=embedding_matrix, **kwargs)
    if key == "bert":
        return build_bert_model(**kwargs)
    raise ValueError('Unknown model name. Use "lstm", "bilstm", or "bert".')
