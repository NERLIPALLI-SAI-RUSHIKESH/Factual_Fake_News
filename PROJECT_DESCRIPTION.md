# Project Description — Fake vs Factual News Detector

## Problem Statement

In the modern digital landscape, misinformation spreads rapidly across social media and news platforms. Distinguishing between fake and factual news manually is time-consuming and often unreliable. This project builds an automated, end-to-end NLP pipeline that classifies a given news article as **Fake** or **Factual** with near-human accuracy.

---

## Objective

To develop a machine learning system that:
1. Ingests raw news datasets
2. Cleans, explores, and preprocesses the text
3. Extracts features using TF-IDF
4. Trains and compares multiple classification models
5. Deploys the best model as a live Streamlit web application

---

## Dataset

- **Source**: Kaggle — Fake and Real News Dataset
- **True.csv**: ~21,000 real news articles sourced from Reuters
- **Fake.csv**: ~23,000 fabricated/fake news articles
- **Labels**: 1 = Factual, 0 = Fake

---

## Methodology

### Step 1 — Data Loading & Labeling
Both datasets are loaded, labeled (0 for Fake, 1 for Factual), and combined into a single shuffled dataframe. Irrelevant columns (`date`, `subject`) are dropped.

### Step 2 — Data Cleaning
- Null value detection and removal
- Duplicate article removal based on the `text` column

### Step 3 — Exploratory Data Analysis
- Class distribution visualization
- Average text length comparison between fake and factual news
- POS (Part-of-Speech) tagging across the corpus using spaCy
- NER (Named Entity Recognition) to identify the top 10 most targeted entities in fake and factual news (Persons, Organizations, Geopolitical Entities)

### Step 4 — Text Preprocessing
A standard NLP preprocessing chain is applied:
1. Title + text concatenation
2. Lowercasing
3. Punctuation and whitespace removal
4. Tokenization (NLTK word_tokenize)
5. Stop word removal (NLTK English stop words)
6. Lemmatization (WordNetLemmatizer)
7. Unigram frequency analysis and plotting for both classes

### Step 5 — Feature Engineering (TF-IDF)
- `TfidfVectorizer` with `max_features=50,000` and `ngram_range=(1, 2)`
- Fitted exclusively on training data to prevent data leakage
- 80/20 stratified train-test split

### Step 6 — Model Training & Comparison
Six models are trained and evaluated on Accuracy and F1-Score:
- Logistic Regression
- Passive Aggressive Classifier
- Multinomial Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Linear SVM (LinearSVC)

The best model is selected by F1-Score and serialized with `joblib`.

### Step 7 — Deployment
A Streamlit web app (`app.py`) allows users to paste any news headline and article text and receive an instant Fake/Factual prediction.

---

## Key Technologies

| Category | Tools / Libraries |
|---|---|
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| NLP | NLTK, spaCy |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Web Application | Streamlit |

---

## Results

The best-performing models consistently achieve **98–99% accuracy and F1-Score** on the test set, demonstrating strong generalization. The Passive Aggressive Classifier and Linear SVM perform particularly well due to the high-dimensional, sparse nature of TF-IDF features.

---

## Future Improvements

- Incorporate transformer-based models (BERT, RoBERTa) for contextual understanding
- Add multi-language support
- Build a browser extension for real-time fact-checking
- Introduce a confidence score display in the Streamlit UI
- Integrate with a live news API for automated monitoring

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK resources
python -m nltk.downloader punkt stopwords wordnet

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run the notebook (generates model/ artifacts)
jupyter notebook fake_vs_factual_news.ipynb

# 5. Launch the web app
streamlit run app.py
```
