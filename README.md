## Sentiment Analysis of Tweets Using Deep Learning

**Course:** CS 171 – Introduction to Machine Learning, San Jose State University  
**Team:** Shresthkumar Karnani, Robel Sebhat

---

## 1. Project Overview

This project builds an end-to-end deep learning pipeline to classify tweets into **three sentiment classes**: **positive**, **neutral**, and **negative**. In addition to sentiment classification, we include an **emotion detection** extension using a pretrained HuggingFace model to annotate tweet emotions (demo/enrichment).

---

## 2. Dataset

- **Dataset:** Kaggle Twitter Tweets Sentiment Dataset  
- **Size:** ~27,481 tweets  
- **Labels:** positive (~31%), neutral (~40%), negative (~28%)  
- **Link:** `https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset`  
- **Expected location:** place the downloaded file at `data/Tweets.csv`

Expected columns include: `textID`, `text`, `selected_text`, `sentiment`.

---

## 3. Project Structure

```
CS171_project/
├── preprocess.py
├── embeddings.py
├── model.py
├── evaluate.py
├── emotion_detection.py
├── test_pipeline.py
├── requirements.txt
├── data/
│   └── Tweets.csv
├── processed/
├── glove/
└── results/
```

---

## 4. Setup

```bash
pip install -r requirements.txt
```

---

## 5. How to Run (in order)

```bash
# Step 1: Preprocess
python3 preprocess.py

# Step 2: GloVe embeddings (optional)
python3 embeddings.py

# Step 3: Emotion detection demo (optional)
python3 emotion_detection.py

# Step 4: Train and evaluate
python3 evaluate.py

# Step 5: Run tests
pytest test_pipeline.py -v
```

---

## 6. Methodology

- **Data preprocessing**
  - Expand common contractions (e.g., "don't" → "do not")
  - Lowercase text
  - Remove URLs, @mentions, and strip `#` while keeping hashtag words
  - Regex tokenization to keep word-like tokens
  - Pad sequences to **maxlen = 30**
- **Feature representation**
  - Keras **Embedding** layer (trainable)
  - Optional **GloVe 6B 100d** initialization via `embeddings.py`
- **Models**
  - **LSTM**: stacked **128 → 64**
  - **BiLSTM**: **Bidirectional(128) → 64**
  - **DistilBERT**: pretrained encoder + lightweight classifier, fine-tuned for **1 epoch** (CPU-friendly demo run)
- **Training**
  - Optimizer: **Adam**
  - Loss: **sparse categorical crossentropy**
  - **Class weights** to mitigate label imbalance
  - **3-fold cross-validation** for LSTM/BiLSTM; final evaluation on held-out test set
- **Emotion detection (extension)**
  - HuggingFace emotion model: `j-hartmann/emotion-english-distilroberta-base` via `transformers` pipeline

---

## 7. Results

| Model | Type | CV Accuracy | Test Accuracy | Test F1 |
|------|------|-------------|---------------|---------|
| LSTM | Trained from scratch | 0.6668 | 0.6910 | 0.6921 |
| BiLSTM | Trained from scratch | 0.6757 | 0.6856 | 0.6859 |
| DistilBERT | Pretrained + fine-tuned | N/A (single run) | 0.6491 | 0.6460 |

**Interpretation:** The **LSTM** achieved the best test accuracy (**69.1%**). The **BiLSTM** was close (**68.6%**). **DistilBERT** reached **64.9%** with only **1 fine-tuning epoch on CPU**; with more fine-tuning epochs (and/or GPU), performance would likely improve. In our limited fine-tuning setting, the from-scratch LSTM/BiLSTM models outperformed BERT.

---

## 8. Key Findings

- Removing stopwords hurt performance — words like **"not"** and **"never"** carry sentiment signal.
- Contraction expansion **before** punctuation stripping significantly cleaned the vocabulary.
- Trainable embeddings outperformed frozen GloVe on informal tweet text.
- Class weighting was essential to prevent collapse to the majority class.

---

## 9. File Descriptions

| File | Description |
|------|-------------|
| `preprocess.py` | Cleans tweets, tokenizes, encodes labels, splits data, pads sequences |
| `embeddings.py` | Downloads GloVe 6B, builds embedding matrix |
| `model.py` | Defines LSTM, BiLSTM, and DistilBERT models |
| `evaluate.py` | 3-fold CV training, final test evaluation, saves results and plots |
| `emotion_detection.py` | Assigns emotion labels using pretrained HuggingFace model |
| `test_pipeline.py` | pytest tests for preprocessing, models, and emotion detection |

---

## 10. References

- P. Nandwani and R. Verma, "A review on sentiment analysis and emotion detection from text," Social Network Analysis and Mining, vol. 11, no. 1, Aug. 2021. doi:10.1007/s13278-021-00776-6
- Kaggle Dataset: `https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset`
- HuggingFace DistilBERT: `https://huggingface.co/distilbert-base-uncased`
- HuggingFace Emotion Model: `https://huggingface.co/j-hartmann/emotion-english-distilroberta-base`
