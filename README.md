# 🎵 Predicting Music Hits with Machine Learning

This project uses machine learning to predict whether a song will be a **hit** (Spotify popularity ≥ 60) based on audio features extracted from the Spotify API, such as danceability, energy, acousticness, loudness, and more.

The main goal is to build a classification model capable of identifying patterns present in successful songs.

## 📊 Dataset

- **Source**: [Song Popularity Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset)
- **File**: `song_data.csv`
- **Key columns**:
  - `song_popularity`: Target variable (0–100)
  - Audio features: `danceability`, `energy`, `loudness`, `acousticness`, `instrumentalness`, `audio_valence`, etc.
- **Created target variable**: `is_hit` = 1 if `song_popularity` ≥ 60, otherwise 0 (~43% hits)

## 🛠️ Requirements

Install the required dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn plotly
