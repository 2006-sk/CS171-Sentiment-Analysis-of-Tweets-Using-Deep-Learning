# CS 171 Tweet Sentiment Analysis

This project builds a sentiment analysis pipeline for a tweet sentiment dataset (positive / neutral / negative).

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download `Tweets.csv` from Kaggle and place it at `data/Tweets.csv`:

`https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset`

Expected columns: `textID`, `text`, `selected_text`, `sentiment`.

## Run preprocessing (Step 1)

Run:

```bash
python preprocess.py
```

Outputs will be saved to `processed/`:
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `tokenizer.json`
- `label_classes.json`
