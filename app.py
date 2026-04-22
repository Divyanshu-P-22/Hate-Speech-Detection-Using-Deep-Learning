import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Page Configuration
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="centered")

# Download required NLTK data for deployment
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Load the Saved Model and Tokenizer
@st.cache_resource 
def load_assets():
    model = tf.keras.models.load_model('hate_speech_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# 3. Preprocessing Functions
punctuations_list = string.punctuation

def remove_punctuations(text):
    text = str(text).lower() # Lowercase first
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    imp_words = []
    
    for word in str(text).split():
        if word not in stop_words:
            imp_words.append(lemmatizer.lemmatize(word))
            
    return " ".join(imp_words)

# 4. Streamlit User Interface
st.title("🚫 Hate Speech Detection System")
st.markdown("Enter a comment or tweet below to check if it contains hateful or offensive content.")

user_input = st.text_area("Input Text:", placeholder="Type here...", height=150)

if st.button("Analyze Content"):
    if user_input.strip() == "":
        st.info("Please enter some text to analyze.")
    else:
        loading_phrases = [
            "Consulting the neural overlords...",
            "Translating internet troll into English...",
            "Scanning for emotional damage...",
            "Surfing the darkest corners of the web...",
            "Math is happening. Please hold...",
            "Asking the GPU nicely to do its job...",
            "Sanitizing inputs with digital soap...",
            "Judging your text silently...",
            "Checking for internet toxicity..."
        ]
        with st.spinner(random.choice(loading_phrases)):
            # Preprocess input
            text_cleaned = remove_punctuations(user_input)
            text_cleaned = remove_stopwords(text_cleaned)
            
            # Tokenize and Pad
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
                st.balloons() # Drops balloons down the screen
                st.caption("Wow, a nice comment on the internet! That's rare.")
                
with st.expander("📝 Developer Disclaimer"):
    st.write("""
    *This model was trained on thousands of highly toxic tweets. As the Project Developer, I take zero responsibility for the emotional damage caused by whatever text you just decided to test it with.* *If the model is wrong, please blame the internet, not the developer.*
    """)

st.divider()
st.caption("Deep Learning Project Submission | Developed by Divyanshu Prajapat")
