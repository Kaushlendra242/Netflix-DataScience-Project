import streamlit as st
import pandas as pd
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

# Files in the same folder
CSV_PATH   = "netflix_cleaned.csv"
VECT_PATH  = "netflix_vectorizer.pkl"
TFIDF_PATH = "netflix_tfidf_matrix.npz"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    vectorizer = joblib.load(VECT_PATH)
    tfidf_matrix = sparse.load_npz(TFIDF_PATH)
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_data()

st.title("📺 Netflix Content Recommender")

query = st.text_input("Enter a movie or TV show title:")
top_n = st.slider("Number of recommendations", 1, 20, 5)

def recommend_by_title(title, top_n=10):
    q = str(title).strip().lower()
    candidates = df[df["title"].str.lower().str.contains(q, na=False)]
    if candidates.empty:
        return pd.DataFrame()
    idx = candidates.index[0]
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    recs = sims.argsort()[::-1]
    recs = [i for i in recs if i != idx][:top_n]
    return df.iloc[recs][["title","type","release_year","listed_in"]]

if st.button("Recommend"):
    if not query:
        st.warning("Please enter a title.")
    else:
        recs = recommend_by_title(query, top_n)
        if recs.empty:
            st.info("No similar titles found.")
        else:
            st.subheader("Recommendations:")
            for _, row in recs.iterrows():
                st.markdown(f"### 🎬 {row['title']}")
                st.write(f"**Type:** {row['type']} | **Year:** {row['release_year']}")
                st.write(f"**Genres:** {row['listed_in']}")
                st.write("---")
