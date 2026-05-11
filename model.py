from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
)
from tensorflow.keras.optimizers import Adam


def _compile(model: Model) -> Model:
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(
    vocab_size: int = 20000,
    embed_dim: int = 128,
    max_len: int = 50,
    num_classes: int = 3,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Model:
    """
    Embedding → LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dense(num_classes, softmax)
    If embedding_matrix is provided, Embedding is frozen with GloVe (100-d).
    """
    if embedding_matrix is not None:
        embed_dim = int(embedding_matrix.shape[1])
        emb = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=False,
        )
    else:
        emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)

    inputs = Input(shape=(max_len,), dtype="int32")
    x = emb(inputs)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return _compile(Model(inputs=inputs, outputs=outputs, name="lstm_sentiment"))


def build_bilstm_model(
    vocab_size: int = 20000,
    embed_dim: int = 128,
    max_len: int = 50,
    num_classes: int = 3,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Model:
    """
    Embedding → Bidirectional LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dense(num_classes, softmax)
    If embedding_matrix is provided, Embedding is frozen with GloVe (100-d).
    """
    if embedding_matrix is not None:
        embed_dim = int(embedding_matrix.shape[1])
        emb = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=False,
        )
    else:
        emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)

    inputs = Input(shape=(max_len,), dtype="int32")
    x = emb(inputs)
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return _compile(Model(inputs=inputs, outputs=outputs, name="bilstm_sentiment"))


def build_bert_model(num_classes: int = 3) -> Model:
    """
    DistilBERT sentiment classifier using HuggingFace transformers.
    - loads distilbert-base-uncased using TFAutoModelForSequenceClassification
    - freezes base weights
    - adds a softmax on top of logits
    """
    from transformers import TFAutoModelForSequenceClassification

    bert = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
    )

    # Freeze base encoder weights, keep classification head trainable.
    if hasattr(bert, "distilbert"):
        bert.distilbert.trainable = False
    else:
        bert.trainable = False

    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    logits = bert({"input_ids": input_ids, "attention_mask": attention_mask}).logits
    probs = tf.keras.layers.Softmax(name="softmax")(logits)

    model = Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=probs,
        name="distilbert_sentiment",
    )
    return _compile(model)


def get_model(
    name: str,
    embedding_matrix: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Model:
    """
    Factory to build sentiment models.
    name: "lstm", "bilstm", or "bert"
    embedding_matrix: optional frozen GloVe weights for lstm/bilstm only.
    """
    key = name.lower().strip()
    if key == "lstm":
        return build_lstm_model(embedding_matrix=embedding_matrix, **kwargs)
    if key == "bilstm":
        return build_bilstm_model(embedding_matrix=embedding_matrix, **kwargs)
    if key == "bert":
        return build_bert_model(**kwargs)
    raise ValueError('Unknown model name. Use "lstm", "bilstm", or "bert".')
