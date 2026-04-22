import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data for deployment
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 1. Load the Saved Model and Tokenizer
# Using st.cache_resource ensures these heavy files are only loaded once, making the app fast
@st.cache_resource 
def load_assets():
    model = tf.keras.models.load_model('hate_speech_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# 2. Replicate Your Exact Training Preprocessing
def preprocess_text(text):
    # Lowercase
    text = str(text).lower()
    
    # Remove punctuations
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    text = text.translate(temp)
    
    # Remove stopwords and lemmatize
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    imp_words = []
    
    for word in text.split():
        if word not in stop_words:
            # I applied a slight correction here: your training code ran the lemmatizer 
            # but appended the original word. This ensures the lemmatized version is kept!
            lemmatized_word = lemmatizer.lemmatize(word)
            imp_words.append(lemmatized_word)
            
    return " ".join(imp_words)

# 3. Build the Streamlit User Interface
st.set_page_config(page_title="Hate Speech Detector", page_icon="🛡️")

st.title("🛡️ Hate Speech Detection Engine")
st.markdown("Enter a tweet or comment below to analyze its content.")

# Input text box
user_input = st.text_area("Text Input", height=150, placeholder="Type something here...")

# 4. Handle the Prediction Logic
if st.button("Analyze Text", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            # Step A: Preprocess the raw input
            cleaned_text = preprocess_text(user_input)
            
            # Step B: Convert to sequence and pad
            # Note: Your training code used maxlen=100 and default padding ('pre')
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=100)
            
            # Step C: Predict
            prediction = model.predict(padded_seq)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            
            # Step D: Map the output to the class labels
            # 0 = Hate Speech, 1 = Offensive Language, 2 = Neither/Neutral
            if predicted_class_index == 0:
                st.error(f"**Result:** Hate Speech detected. (Confidence: {confidence:.2f}%)")
            elif predicted_class_index == 1:
                st.warning(f"**Result:** Offensive Language detected. (Confidence: {confidence:.2f}%)")
            elif predicted_class_index == 2:
                st.success(f"**Result:** Neutral / Neither detected. (Confidence: {confidence:.2f}%)")
                
            # Optional: Display the cleaned text so you can debug what the model actually "saw"
            with st.expander("View Preprocessing Details"):
                st.write("**Original:**", user_input)
                st.write("**Cleaned & Lemmatized:**", cleaned_text)
