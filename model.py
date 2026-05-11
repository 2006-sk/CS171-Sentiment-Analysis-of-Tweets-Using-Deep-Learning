from __future__ import annotations

from typing import Any

import tensorflow as tf

Model = tf.keras.Model
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
Input = tf.keras.layers.Input
Adam = tf.keras.optimizers.Adam


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
) -> Model:
    """
    Embedding → LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dense(num_classes, softmax)
    """
    inputs = Input(shape=(max_len,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(
        inputs
    )
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
) -> Model:
    """
    Embedding → Bidirectional LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dense(num_classes, softmax)
    """
    inputs = Input(shape=(max_len,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(
        inputs
    )
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


def get_model(name: str, **kwargs: Any) -> Model:
    """
    Factory to build sentiment models.
    name: "lstm", "bilstm", or "bert"
    """
    key = name.lower().strip()
    if key == "lstm":
        return build_lstm_model(**kwargs)
    if key == "bilstm":
        return build_bilstm_model(**kwargs)
    if key == "bert":
        return build_bert_model(**kwargs)
    raise ValueError('Unknown model name. Use "lstm", "bilstm", or "bert".')
