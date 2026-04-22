---

# 🚫 Hate Speech Detection using Deep Learning

This repository contains a Deep Learning project designed to classify text into three categories: **Hate Speech**, **Offensive Language**, or **Neutral**. The project features a trained **Bidirectional LSTM** model and is deployed as a live web application using **Streamlit**.

## 🌐 Live Demo
You can access the deployed web application here: 
`[INSERT YOUR STREAMLIT URL HERE]`

## 📊 Project Overview
The goal of this project is to automate the detection of harmful content in social media comments or tweets. By leveraging Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks, the model understands the context of sentences to provide accurate classifications.

### Key Features:
* **Preprocessing Pipeline:** Text cleaning, punctuation removal, stopword filtering, and Lemmatization.
* **Balanced Dataset:** Handled class imbalance using sampling techniques to improve model fairness.
* **Deep Learning Architecture:** Utilizes Bidirectional LSTMs for capturing long-term dependencies in text.
* **Real-time Deployment:** A user-friendly interface for instant text analysis.

## 🏗️ Model Architecture
The model is built using TensorFlow/Keras with the following layers:
1.  **Embedding Layer:** Maps words to dense vectors.
2.  **Bidirectional LSTM:** Processes sequences in both directions to capture better context.
3.  **Dense Layers:** Deep neural layers with ReLU activation and L1 Regularization.
4.  **Batch Normalization & Dropout:** Layers added to prevent overfitting and ensure stable training.
5.  **Softmax Output:** Predicts the probability of the three target classes.


## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **NLP Tools:** NLTK (Stopwords, WordNetLemmatizer)
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

## 📁 Repository Structure
```text
├── app.py                # Streamlit web application script
├── requirements.txt      # List of dependencies for deployment
├── hate_speech_model.h5  # Trained LSTM model
├── tokenizer.pkl         # Saved tokenizer for text-to-sequence conversion
└── README.md             # Project documentation
```

## 🚀 How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📈 Performance
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Early Stopping:** Implemented to restore best weights during training.

---
**Developed by:** Divyanshu Prajapat  
**Submission:** Deployed ML/DL Project Submission
---
