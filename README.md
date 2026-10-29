# AI Sentiment Analyzer

A beginner-level NLP project that classifies a short review as positive or negative using Python, Streamlit, TF-IDF, and Logistic Regression.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. (Optional) Create and activate a virtual environment:
   `python -m venv venv`
   Windows: `venv\Scripts\activate`
4. Install packages:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`

## What I learned
- Natural Language Processing (NLP) basics
- Converting text into numeric features with TF-IDF
- Training a binary text classifier with Logistic Regression
- Creating an interactive Streamlit application

## Important limitation
The built-in dataset is intentionally small and is for practice only. Predictions are not reliable enough for business decisions. Improve it with a larger, balanced, representative dataset and proper evaluation before deployment.
