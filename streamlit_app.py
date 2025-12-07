# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="Netflix Recommendation System", layout="wide")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
CSV_PATH = "netflix_cleaned.csv"
VECTORIZER_PATH = "netflix_vectorizer.pkl"
TFIDF_MATRIX_PATH = "netflix_tfidf_matrix.npz"

# ---------------------------------------------------------
# Load data & models
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)

@st.cache_resource
def load_tfidf_matrix():
    return load_npz(TFIDF_MATRIX_PATH)

df = load_data()
vectorizer = load_vectorizer()
tfidf_matrix = load_tfidf_matrix()

# ---------------------------------------------------------
# Recommendation Logic
# ---------------------------------------------------------
def recommend_movies(title, df, vectorizer, tfidf_matrix):
    title = title.lower()

    matches = df[df['title'].str.lower() == title]
    if matches.empty:
        return None, "Title not found in the dataset."

    idx = matches.index[0]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    top_indices = sim_scores.argsort()[-6:][::-1][1:]

    recommendations = df.iloc[top_indices][['title', 'type', 'listed_in', 'release_year', 'rating']]
    return recommendations, None

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🎬 Netflix Recommendation System")
st.markdown("Enter a movie/TV show title to get similar recommendations!")

st.sidebar.header("About This App")
st.sidebar.write("""
**Netflix ML Project**
- TF-IDF NLP-based recommendation  
- Cosine similarity  
- Clean UI built with Streamlit  
""")

# ---------------------------------------------------------
# Search Box
# ---------------------------------------------------------
title_input = st.text_input("Enter a Movie/TV Show Title")

if st.button("Get Recommendations"):
    if not title_input.strip():
        st.warning("Please enter a valid title.")
    else:
        with st.spinner("Analyzing..."):
            recommendations, error = recommend_movies(title_input, df, vectorizer, tfidf_matrix)

        if error:
            st.error(error)
        else:
            st.success("Top Recommendations:")

            for _, row in recommendations.iterrows():
                st.markdown(f"""
                ### 🎥 {row['title']}
                - **Type:** {row['type']}
                - **Genre:** {row['listed_in']}
                - **Release Year:** {row['release_year']}
                - **Rating:** {row['rating']}
                ---
                """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by **Kaushlendra Pratap Singh**")
