---
title: IMDb Sentiment (Deep Learning)
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# IMDb Sentiment Classifier — live demo

Interactive demo of a Word2Vec-embedding neural network trained from scratch on
25,000 IMDb movie reviews (**83.8% test accuracy**). Type a movie review and get
a live positive/negative prediction.

The model (`sentiment_pool.keras`) was selected by an architecture bake-off
(global average pooling beat a Flatten-MLP and an LSTM) over leakage-free
train/val/test splits.

**Full project & training pipeline:**
https://github.com/Inspiring-Ming/imdb-sentiment-dl

This Space mirrors the `demo/` folder of that repo.
