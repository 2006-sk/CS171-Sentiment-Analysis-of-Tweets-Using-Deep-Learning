from __future__ import annotations

import os
from typing import List

import pandas as pd
from transformers import pipeline


VALID_EMOTIONS = {
    "joy",
    "anger",
    "sadness",
    "fear",
    "disgust",
    "surprise",
    "neutral",
}


def load_emotion_model():
    """
    Load the pretrained HuggingFace emotion detection pipeline.
    """
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
    )


def assign_emotions(texts: List[str]) -> List[str]:
    """
    Run the emotion pipeline in batches of 64 and return a list of emotion labels.
    """
    clf = load_emotion_model()

    out: List[str] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = clf(batch)
        # With top_k=1, each element is a list with 1 dict: [{"label":..., "score":...}]
        for p in preds:
            label = p[0]["label"]
            out.append(label)
    return out


def add_emotion_labels(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add an 'emotion' column to df based on the specified text column.
    """
    texts = df[text_col].astype(str).tolist()
    print(f"Assigning emotions for {len(texts)} rows...")
    emotions = assign_emotions(texts)
    df = df.copy()
    df["emotion"] = emotions
    return df


if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)
    df = pd.read_csv("data/Tweets.csv").head(200)
    df = add_emotion_labels(df, text_col="text")
    print("Emotion distribution:")
    print(df["emotion"].value_counts())
    out_path = "processed/sample_with_emotions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

