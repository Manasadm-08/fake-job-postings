import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# PAGE CONFIG
st.set_page_config(page_title="Fake Job Detection", page_icon="🧠")

# LOAD DATA
df = pd.read_csv("fake_job_postings.csv")
df.fillna('', inplace=True)

# Combine text
df['text'] = df['title'] + " " + df['description'] + " " + df['requirements']

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text.lower()

df['text'] = df['text'].apply(clean_text)


# MODEL
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']

model = LogisticRegression(max_iter=200, C=0.5)
model.fit(X, y)


# UI DESIGN
st.title("🧠 Fake Job Detection System")
st.markdown("### 🔍 Check whether a job posting is **Fake or Real**")
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "This app uses Machine Learning (TF-IDF + Logistic Regression) "
    "to detect fake job postings."
)

# Input
user_input = st.text_area("📄 Enter Job Description:")

# Example button
if st.button("Try Example"):
    user_input = "Earn money fast from home no experience needed"

# Predict
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])

        with st.spinner("Analyzing..."):
            result = model.predict(input_vec)
            proba = model.predict_proba(input_vec)

        # Show input
        st.write("### 📝 Your Input:")
        st.write(user_input)

        # Show result
        if result[0] == 1:
            st.error("⚠ This looks like a Fake Job")
        else:
            st.success("✅ This looks like a Real Job")

        # Confidence score
        confidence = round(max(proba[0]) * 100, 2)
        st.write(f"### 📊 Confidence: {confidence}%")
