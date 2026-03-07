import streamlit as st
import pickle
import re
import string
import nltk
import numpy as np
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download resources
nltk.download('stopwords')
nltk.download('wordnet')

# load model
model = pickle.load(open("fake_job_svm.pkl","rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# text cleaning
def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r'\d+',"",text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# page config
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🧠",
    layout="wide"
)

# custom css
st.markdown("""
<style>

body{
background:#0e1117;
}

.title{
font-size:48px;
font-weight:700;
text-align:center;
color:white;
}

.subtitle{
text-align:center;
color:#9aa0a6;
font-size:18px;
margin-bottom:40px;
}

.card{
background:#161b22;
padding:30px;
border-radius:12px;
box-shadow:0px 6px 20px rgba(0,0,0,0.3);
}

.result-real{
background:#133a1b;
padding:20px;
border-radius:10px;
text-align:center;
font-size:24px;
font-weight:bold;
color:#7CFC98;
}

.result-fake{
background:#3b0d0d;
padding:20px;
border-radius:10px;
text-align:center;
font-size:24px;
font-weight:bold;
color:#ff6b6b;
}

</style>
""", unsafe_allow_html=True)

# sidebar
st.sidebar.title("🧠 Model Information")

st.sidebar.write("Model: TF-IDF + SVM")
st.sidebar.write("Accuracy: ~95%")

st.sidebar.write("---")

st.sidebar.write("### Example Job")

if st.sidebar.button("Load Fake Job Example"):
    st.session_state["example"] = """
Earn $4000 per week from home.
No experience required.
Limited positions available.
Apply now!
"""

# title
st.markdown('<p class="title">AI Fake Job Detection</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Detect fraudulent job postings using Machine Learning</p>',
unsafe_allow_html=True
)

# layout
col1, col2 = st.columns([3,2])

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    job_text = st.text_area(
        "Paste Job Description",
        value=st.session_state.get("example",""),
        height=250,
        placeholder="Paste full job description here..."
    )

    analyze = st.button("🔎 Analyze Job Posting")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("How it Works")

    st.write("""
1. Job text is cleaned using NLP  
2. TF-IDF converts text to features  
3. SVM model predicts job authenticity  
4. Confidence score is generated
""")

    st.markdown('</div>', unsafe_allow_html=True)


# prediction
if analyze and job_text.strip() != "":

    with st.spinner("AI is analyzing the job posting..."):
        time.sleep(1.5)

    cleaned = clean_text(job_text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    # UPDATED CONFIDENCE CALCULATION
    proba = model.predict_proba(vector)[0]

    fake_prob = proba[1]
    real_prob = proba[0]

    confidence = max(fake_prob, real_prob)

    st.write("")
    st.write("### AI Detection Result")

    if prediction[0] == 1:

        st.markdown(
        '<div class="result-fake">⚠ Fake Job Posting Detected</div>',
        unsafe_allow_html=True
        )

        st.write("Fake Probability:", round(fake_prob*100,2), "%")

    else:

        st.markdown(
        '<div class="result-real">✅ Legitimate Job Posting</div>',
        unsafe_allow_html=True
        )

        st.write("Real Probability:", round(real_prob*100,2), "%")

    st.write("")

    st.progress(confidence)

    st.write("Confidence Score:", round(confidence*100,2), "%")


st.write("")
st.write("---")

st.caption("Fake Job Detection System • Machine Learning Project")