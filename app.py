import streamlit as st
import tensorflow as tf
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="💼",
    layout="centered"
)

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
model = tf.keras.models.load_model("fake_job_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200


# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


# -------------------------------
# Prediction Function
# -------------------------------
def predict_job(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)[0][0]

    return pred


# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #2c3e50;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.real {
    background-color: #d4edda;
    color: #155724;
}

.fake {
    background-color: #f8d7da;
    color: #721c24;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# Title Section
# -------------------------------
st.markdown('<p class="main-title">💼 Fake Job Detection System</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="sub-title">Detect fraudulent job postings using NLP and Deep Learning (Bidirectional LSTM)</p>',
    unsafe_allow_html=True
)

st.write("")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📄 Enter Job Description")

job_text = st.text_area(
    "",
    height=200,
    placeholder="Paste the job description here..."
)

st.write("")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Analyze Job Posting"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")

    else:

        prediction = predict_job(job_text)

        probability = prediction * 100

        if prediction > 0.35:

            st.markdown(
                f'<div class="result-box fake">🚨 Fake Job Detected<br>Confidence: {probability:.2f}%</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="result-box real">✅ Legitimate Job Posting<br>Confidence: {100-probability:.2f}%</div>',
                unsafe_allow_html=True
            )


# -------------------------------
# Divider
# -------------------------------
st.write("---")

# -------------------------------
# About Section
# -------------------------------
st.subheader("📊 About This Project")

st.write("""
This system detects **fraudulent job postings** using **Natural Language Processing and Deep Learning**.

Model used:
- Bidirectional LSTM
- Text Tokenization
- Sequence Padding

Accuracy Achieved:
**~97%**
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    '<p class="footer">Built using Python, TensorFlow, NLP and Streamlit</p>',
    unsafe_allow_html=True
)