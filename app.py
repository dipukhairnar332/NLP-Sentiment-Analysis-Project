import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt

# ----------------------------
# Load TF-IDF vectorizer and Logistic Regression model
# ----------------------------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(s):
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)              # remove extra spaces
    s = re.sub(r'[^a-z0-9\s]', '', s)       # keep letters & numbers
    return s

# ----------------------------
# Store prediction history in session_state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("📊 Sentiment Analysis App")
st.write("Enter a review or sentence below to predict its sentiment.")

user_input = st.text_area("✍️ Enter your text:")

if st.button("Predict"):
    if user_input.strip():
        # Preprocess input
        text = clean_text(user_input)
        X_input = vectorizer.transform([text])

        # Predict sentiment
        prediction = logreg_model.predict(X_input)[0]
        proba = logreg_model.predict_proba(X_input)[0]  # probability scores

        # ----------------------------
        # Display prediction
        # ----------------------------
        st.subheader("🔎 Prediction:")
        if prediction.lower() == "positive":
            st.success("🙂 Positive Sentiment")
        elif prediction.lower() == "negative":
            st.error("😠 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")

        # ----------------------------
        # Save history
        # ----------------------------
        st.session_state.history.append((user_input, prediction))

        # ----------------------------
        # Show probabilities as bar chart
        # ----------------------------
        st.subheader("📊 Prediction Confidence")
        color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
        colors = [color_map.get(cls.lower(), "blue") for cls in logreg_model.classes_]

        fig, ax = plt.subplots()
        ax.bar(logreg_model.classes_, proba, color=colors)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Sentiment Confidence")
        st.pyplot(fig)

        # ----------------------------
        # Show text statistics
        # ----------------------------
        st.subheader("📌 Text Statistics")
        st.write("Word count:", len(text.split()))
        st.write("Character count:", len(text))

    else:
        st.warning("⚠️ Please enter some text.")

# ----------------------------
# Show prediction history
# ----------------------------
if st.session_state.history:
    st.subheader("📝 Prediction History")
    for i, (txt, pred) in enumerate(st.session_state.history, 1):
        st.write(f"{i}. **{txt}** → {pred}")
