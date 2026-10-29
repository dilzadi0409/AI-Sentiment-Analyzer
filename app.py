import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬")
st.title("💬 AI Sentiment Analyzer")
st.write("Type a short review and the machine-learning model will estimate its sentiment.")

positive = [
    "I absolutely love this product", "The service was excellent and friendly",
    "This is a wonderful experience", "The quality is amazing",
    "I am very happy with my purchase", "The app is easy and useful",
    "Fantastic work I highly recommend it", "Everything was perfect",
    "The delivery was fast and smooth", "Great value for the money",
    "I enjoyed using this website", "The staff were very helpful",
    "This product works really well", "A brilliant and lovely experience",
    "I am satisfied with the results", "The design looks beautiful",
    "Very good support and great quality", "It exceeded my expectations",
    "I would happily buy this again", "Excellent experience from start to finish"
]
negative = [
    "I hate this product", "The service was terrible",
    "This was a very bad experience", "The quality is poor",
    "I am unhappy with my purchase", "The app is confusing and slow",
    "Very disappointing I do not recommend it", "Everything went wrong",
    "The delivery was late and damaged", "A waste of money",
    "I dislike using this website", "The staff were rude",
    "This product does not work", "A horrible and frustrating experience",
    "I am not satisfied with the results", "The design looks ugly",
    "Bad support and poor quality", "It failed to meet expectations",
    "I would never buy this again", "Terrible experience from start to finish"
]
texts = positive + negative
labels = [1]*len(positive) + [0]*len(negative)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(texts, labels)

review = st.text_area("Enter a review", placeholder="Example: The product quality is excellent and I love it.")
if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        confidence = max(probabilities) * 100
        if prediction == 1:
            st.success(f"Sentiment: Positive (demo confidence: {confidence:.1f}%)")
        else:
            st.error(f"Sentiment: Negative (demo confidence: {confidence:.1f}%)")
        st.caption("Educational demo only. This small training dataset does not cover sarcasm, mixed opinions, or every writing style.")

with st.expander("How does this project work?"):
    st.write("The app transforms review text using TF-IDF and uses Logistic Regression trained on positive and negative examples to estimate sentiment.")
