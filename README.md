# 🎵 Predicting Music Hits with Machine Learning

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Project%20Completed-success.svg)]()

## 📝 Project Overview
This project builds a **classification model** to predict whether a song will become a **hit** (Spotify popularity ≥ 60) using audio features provided by the Spotify API.

Key features include `danceability`, `energy`, `acousticness`, `loudness`, `instrumentalness`, `valence`, and more. The target variable `is_hit` is binary (1 = hit, 0 = non-hit), with ~43% hits in the dataset.

The pipeline compares **Random Forest** (best performer) against **Logistic Regression** and includes feature selection, scaling, EDA, and model evaluation.

## 🎯 Key Results

| Metric                  | Random Forest | Logistic Regression | Notes                          |
|-------------------------|---------------|---------------------|--------------------------------|
| **Test Accuracy**       | **78%**       | 62%                 | RF significantly outperforms   |
| **Cross-Validation Avg**| ~77%          | -                   | Good generalization            |
| **Training Accuracy**   | 99%           | -                   | Mild overfitting detected      |

- **Top 5 Features** (ANOVA F-test): `acousticness`, `danceability`, `energy`, `instrumentalness`, `loudness`
- **Most Important Features** (Random Forest): Energy, danceability, loudness, valence lead the way

## 🚀 Technical Pipeline
Structured end-to-end workflow:

1. **Data Loading & Exploration**  
   Loaded dataset and created binary target `is_hit`.

2. **EDA**  
   Visualized class distribution, correlations with hit status, and boxplots (e.g., danceability by hit).

3. **Preprocessing**  
   - Dropped non-feature columns  
   - Standardized features with `StandardScaler`

4. **Feature Selection**  
   Used `SelectKBest` with ANOVA F-test to identify top 5 predictors.

5. **Modeling**  
   - Random Forest (class_weight='balanced') → best results  
   - Logistic Regression → baseline comparison

6. **Evaluation**  
   Classification report, confusion matrix, accuracy, cross-validation, and feature importance plots.

7. **Predictions Export**  
   Generated `hits_predictions.csv` with song names, predicted hit status, and probability.

## 💡 Key Insights
- Hit songs tend to have **higher danceability**, **higher energy**, and **lower acousticness**.
- **Loudness** and **valence** also play strong roles in popularity.
- Random Forest captures non-linear relationships far better than Logistic Regression.
- Slight overfitting observed (99% train vs 78% test) — room for hyperparameter tuning.

## 🛠️ Tech Stack
- **Language**: Python
- **Core Libraries**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (RandomForestClassifier, LogisticRegression, StandardScaler, SelectKBest)
- **Visualization**: Matplotlib, Seaborn, Plotly (interactive plots)

## 📊 Dataset Access
- **Source**: Song Popularity Dataset
- **Download Link**: [Kaggle - Song Popularity Dataset](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset)
- **Main File**: `song_data.csv` (~18k songs)

**How to Load in Python:**
```python
import pandas as pd

# Recommended: place in /data folder
df = pd.read_csv('data/song_data.csv')

# Or in repository root
# df = pd.read_csv('song_data.csv')
