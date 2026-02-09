import streamlit as st
import joblib
import re
import string

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Header
st.title("📰 Fake News Detection System")
st.markdown(
    "Detect whether a news article is **Fake** or **Real** using "
    "**Logistic Regression** and **Machine Learning**."
)

st.divider()

# Input section
news_text = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste the full news article here..."
)

# Prediction button
if st.button("Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        cleaned_text = clean_text(news_text)
        news_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(news_vector)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

st.divider()

# Extra info section
with st.expander("📘 About This Project"):
    st.write("""
    - **Algorithm:** Logistic Regression  
    - **Feature Extraction:** TF-IDF  
    - **Dataset:** Kaggle Fake & True News  
    - **Accuracy:** ~98%  
    - **Platform:** Python + Streamlit
    """)

# Footer
st.caption("© Fake News Detection | Machine Learning Project")
