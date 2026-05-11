# CS 171 Tweet Sentiment Analysis

End-to-end pipeline for **three-class tweet sentiment** (negative / neutral / positive): preprocessing, **GloVe** embeddings, optional **emotion** labels, and models (**LSTM**, **BiLSTM** with frozen GloVe, **DistilBERT**).

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Dataset

Download `Tweets.csv` from Kaggle and save it as `data/Tweets.csv`:

[https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

Expected columns include: `textID`, `text`, `selected_text`, `sentiment`.

## 3. Preprocessing

Builds cleaned text, stratified **80% / 10% / 10%** train/validation/test splits, Keras tokenizer sequences (length 50), and label encodings under `processed/`:

```bash
python preprocess.py
```

Outputs include `X_*.npy`, `y_*.npy`, `tokenizer.json`, and `label_classes.json`.

## 4. GloVe embeddings

Downloads **GloVe 6B** (100-dimensional vectors) if needed, then builds `processed/embedding_matrix.npy` aligned with the Keras tokenizer (row *i* = vector for index *i*; OOV rows are zeros):

```bash
python embeddings.py
```

## 5. Emotion detection (optional enrichment)

Runs a small Hugging Face emotion classifier on a sample of tweets and writes `processed/sample_with_emotions.csv`:

```bash
python emotion_detection.py
```

## 6. Training and evaluation

Trains with **5-fold** stratified CV (**15** epochs for LSTM/BiLSTM, **3** for BERT), **EarlyStopping** and **ModelCheckpoint** per fold, then evaluates **all three** models on the held-out test set:

```bash
python evaluate.py
```

**Requirements:** run steps 3–4 first so `processed/*.npy`, `tokenizer.json`, and `embedding_matrix.npy` exist.

## 7. Results (`results/`)

After `evaluate.py`:

| Output | Description |
|--------|-------------|
| `model_metrics.csv` | Per-fold **accuracy** and **F1** for **lstm**, **bilstm**, and **bert** |
| `model_metrics_summary.csv` | Per-model **mean ± std** of accuracy and F1 across folds |
| `final_results.csv` | Test **accuracy**, **weighted F1**, and full **classification report** for each model |
| `*_fold_*_accuracy_loss_curve.png` | Side-by-side **train/val accuracy** and **train/val loss** per fold |
| `*_fold_*_cm.png` / `*_test_cm.png` | Confusion matrices (validation folds and final test) |
| `checkpoints/*.keras` | Best weights per fold and per model final run (by `val_loss`) |

## 8. Tests

```bash
pytest test_pipeline.py -v
```

Run **`python preprocess.py`** and **`python embeddings.py`** first so processed artifacts and `embedding_matrix.npy` exist for shape checks.
