import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="AI Translator")
st.title("Simple AI Translator")
st.image(
    "https://placehold.co/800x200/003366/FFFFFF?text=AI%20Translator%3A%20EN%20%E2%86%92%20DE&font=roboto",
    use_column_width=True,
)

st.info(
    """**Welcome!**
This app can:
1) translate text from English to German,
2) analyze the sentiment of English text.
Choose a mode and enter your text."""
)

# Model cache
@st.cache_resource
def load_models():
    return {
        "sentiment": pipeline("sentiment-analysis"),
        "translation": pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de"),
    }

with st.spinner("Loading models (first run)..."):
    models = load_models()
st.success("Models loaded.")

# UI
option = st.selectbox(
    "What do you want to do?",
    ["— select —", "Translation (EN ➔ DE)", "Sentiment analysis (EN)"],
)

# Translation
if option == "Translation (EN ➔ DE)":
    txt = st.text_area("Text in English:", height=150, placeholder="Type text...")
    if st.button("Translate"):
        if txt.strip():
            with st.spinner("Translating..."):
                res = models["translation"](txt)
            st.subheader("Result")
            st.write(f"**Original (EN):** {txt}")
            st.write(f"**Translation (DE):** {res[0]['translation_text']}")
        else:
            st.warning("Please enter text to translate.")

# Sentiment analysis
elif option == "Sentiment analysis (EN)":
    txt = st.text_area("Text in English:", height=150, placeholder="Type text...")
    if st.button("Analyze"):
        if txt.strip():
            with st.spinner("Analyzing..."):
                res = models["sentiment"](txt)
            label = "Positive" if res[0]["label"] == "POSITIVE" else "Negative"
            score = f"{res[0]['score']*100:.2f}%"
            st.subheader("Result")
            st.write(f"**Text:** {txt}")
            st.write(f"**Sentiment:** {label} ({score})")
            with st.expander("Show raw output"):
                st.json(res)
        else:
            st.warning("Please enter text for analysis.")

# Default
else:
    st.write("Choose one option to begin.")

st.divider()
st.caption("Author: 123456")
