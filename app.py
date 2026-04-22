import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

@st.cache_resource
def load_assets():
    # Force the legacy loading if possible
    model = tf.keras.models.load_model('hate_speech_model.h5', compile=False)
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# 1. Page Configuration
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="centered")

# 2. Load the Model and Tokenizer
@st.cache_resource
def load_assets():
    # Ensure these filenames match exactly what you uploaded to GitHub
    model = tf.keras.models.load_model('hate_speech_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# 3. Preprocessing Functions (Identical to your Notebook)
def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return str(text).translate(temp)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = str(text).lower().split()
    imp_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(imp_words)

# 4. Streamlit User Interface
st.title("🚫 Hate Speech Detection System")
st.markdown("Enter a comment or tweet below to check if it contains hateful or offensive content.")

user_input = st.text_area("Input Text:", placeholder="Type here...", height=150)

if st.button("Analyze Content"):
    if user_input.strip() == "":
        st.info("Please enter some text to analyze.")
    else:
        # Preprocess input
        text_cleaned = remove_punctuations(user_input)
        text_cleaned = remove_stopwords(text_cleaned)
        
        # Tokenize and Pad (Using max_len=100 as per your model build)
        seq = tokenizer.texts_to_sequences([text_cleaned])
        padded = pad_sequences(seq, maxlen=100)
        
        # Prediction
        prediction = model.predict(padded)
        class_idx = np.argmax(prediction)
        
        # Map indices to labels
        # 0: Hate Speech, 1: Offensive, 2: Neither
        labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither (Neutral)"}
        result = labels[class_idx]
        
        # Display results with styling
        st.subheader("Result:")
        if class_idx == 0:
            st.error(f"⚠️ {result}")
        elif class_idx == 1:
            st.warning(f"🔔 {result}")
        else:
            st.success(f"✅ {result}")

st.divider()
st.caption("Final Year Project Submission | Developed by Divyanshu Prajapat")