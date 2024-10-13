import streamlit as st
import nltk
from nltk.corpus import stopwords
import joblib

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)

# Summarization function with dynamic levels
def summarize(text, model, tfidf, level='Medium'):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = [preprocess(sentence) for sentence in sentences]
    
    # Transform sentences into TF-IDF vectors
    sentence_vectors = tfidf.transform(cleaned_sentences).toarray()
    
    # Predict importance of each sentence
    sentence_scores = model.predict_proba(sentence_vectors)[:, 1]
    
    # Rank sentences by importance
    ranked_sentences = sorted(((score, sent) for score, sent in zip(sentence_scores, sentences)), reverse=True)
    
    # Determine the number of sentences to include based on the chosen summarization level
    if level == 'Low':
        num_sentences = max(1, int(0.1 * len(sentences)))  # 10% of the sentences
    elif level == 'Medium':
        num_sentences = max(1, int(0.3 * len(sentences)))  # 30% of the sentences
    elif level == 'High':
        num_sentences = max(1, int(0.5 * len(sentences)))  # 50% of the sentences
    
    # Generate summary using the top-ranked sentences
    summary = " ".join([sent for score, sent in ranked_sentences[:num_sentences]])
    
    return summary

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("summarization_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

# Streamlit app
st.title("Text Summarization App")

# Load the model and vectorizer
model, tfidf = load_model()

# Text input
text_input = st.text_area("Enter the text you want to summarize:", height=200)

# Summarization level
summarization_level = st.radio("Choose summarization level:", ("Low", "Medium", "High"), index=1)

# Summarize button
if st.button("Summarize"):
    if not text_input:
        st.error("Please enter some text to summarize.")
    else:
        # Generate summary
        summary = summarize(text_input, model, tfidf, summarization_level)
        st.subheader(f"Generated Summary ({summarization_level} Level):")
        st.write(summary)

# Instructions for running the Streamlit app
st.sidebar.header("How to run this app")
st.sidebar.markdown(
    """
    1. Make sure you have the following files in the same directory:
       - This Streamlit app script
       - `summarization_model.pkl`
       - `tfidf_vectorizer.pkl`
    2. Open a terminal and navigate to the directory containing these files.
    3. Run the command: `streamlit run app.py` (replace `app.py` with the name of this script)
    4. The app should open in your default web browser.
    """
)