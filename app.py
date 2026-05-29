import re
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake vs Factual News Detector",
    page_icon="📰",
    layout="centered",
)

# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model/best_model.pkl")
    tfidf = joblib.load("model/tfidf.pkl")
    with open("model/best_model_name.txt") as f:
        name = f.read().strip()
    return model, tfidf, name

model, tfidf, model_name = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📰 Fake vs Factual News Detector")
st.markdown(f"*Powered by **{model_name}** · TF-IDF (1-2 grams, 50k features)*")
st.markdown("---")

title = st.text_input("📌 Headline", placeholder="Enter the news headline...")
text  = st.text_area("📄 Article Text", height=220,
                      placeholder="Paste the article body here...")

if st.button("🔍 Analyze", use_container_width=True):
    combined = f"{title} {text}".strip()
    if not combined:
        st.warning("⚠️ Please enter a headline or article text before analyzing.")
    else:
        # Preprocess (mirror notebook steps)
        cleaned = re.sub(r"\s+", " ",
                  re.sub(r"[^a-zA-Z\s]", " ", combined.lower()))
        vec  = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ FACTUAL NEWS — This article appears to be **real**.")
        else:
            st.error("❌ FAKE NEWS — This article appears to be **fabricated**.")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and may not be 100% accurate.")
