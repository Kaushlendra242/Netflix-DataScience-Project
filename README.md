**📺 Netflix Data Science Project – End-to-End Analysis, ML & Recommendation System**


**Author: Kaushlendra Pratap Singh
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLP (TF-IDF), Streamlit**

**🚀 Live Streamlit App**

Try the live recommendation system here:

👉 🔗 Streamlit App:
https://netflix-datascience-project-datascientist-kpsingh.streamlit.app/


**🌟 Project Overview**

This project presents a complete end-to-end data science workflow using the Netflix Titles Dataset.
It covers:

✔ Data Cleaning
✔ Exploratory Data Analysis (EDA)
✔ Feature Engineering
✔ Machine Learning (Classification)
✔ NLP-based Recommendation System
✔ Deployment using Streamlit App

The project provides clear insights into Netflix’s content trends and includes production-ready ML components.

**📂 Project Structure**


📁 Netflix-Project
|
│─ 📄 Netflix_data_project.ipynb        # Main Jupyter Notebook
│─ 📄 netflix1.csv                      # Raw dataset
│─ 📄 netflix_cleaned.csv               # Cleaned dataset
│─ 📄 netflix_vectorizer.pkl            # Saved TF-IDF vectorizer
│─ 📄 netflix_tfidf_matrix.npz          # Sparse matrix for recommendations
│─ 📄 streamlit_app.py                  # Deployment app
│─ 📄 README.md                         # Project documentation
│─ 📄 reports/                          # Word/PDF reports



**🧹 1. Data Cleaning & Preprocessing**

Cleaning steps included:

- Duplicate removal

- Handling missing values

- Converting date fields → datetime

- Parsing duration → minutes/seasons

- Extracting primary_country

- Splitting listed_in into genre lists

- Creating derived features

- Building text_all (title + description + cast + director + genre)

✔ Final cleaned dataset saved as: netflix_cleaned.csv


**🔍 2. Exploratory Data Analysis Highlights**


**Content Type Distribution**

- Movies: ~70%

- TV Shows: ~30%

**Most Common Ratings**

- TV-MA

- TV-14

- TV-PG

**Top Content-Producing Countries**

- United States

- India

- United Kingdom

**Popular Genres**

- Dramas

- Comedies

- International Movies

- Documentaries

**Trend Insights**

- Sharp rise in new releases between 2015–2020.

EDA included visualization of:

- Histograms

- Bar charts

- WordClouds

- Time-series plots

**🤖 3. Machine Learning Model**

**Task:**

Predict whether a title is a Movie or TV Show.

**Features Used**

- Release year

- Duration (minutes/seasons)

- Number of genres

- Rating (encoded)

- Primary country (encoded)

**Models Implemented**

- Logistic Regression

- Random Forest

- Hyperparameter Tuning (GridSearchCV)

**Performance Metrics**

- Accuracy

- Precision, Recall, F1-score

- Confusion Matrix
  

**🧠 4. NLP-Based Recommendation System**

A content-based recommendation engine using:

**Techniques**

- TF-IDF vectorization

- Cosine similarity

**Inputs Used**

text_all = title + description + genres + cast + director

**Outputs**

- Top-N recommended titles similar to a selected show/movie

Artifacts saved for deployment:

- netflix_vectorizer.pkl

- netflix_tfidf_matrix.npz

**🚀 5. Streamlit Deployment**

**Features**

- Search for any Netflix title

- Get top recommendations instantly

- Lightweight & user-friendly web UI

**Run locally**

streamlit run streamlit_app.py

**📈 6. Key Insights from the Project**

- Netflix heavily features drama, comedy, and international content.

- Strong growth in content acquisition after 2015.

- Most titles are rated for mature audiences (TV-MA).

- The ML model successfully differentiates Movies vs TV Shows.

- Recommendation system provides meaningful and highly relevant suggestions.

**📝 7. Conclusion**

This project showcases expertise in:

- End-to-end data science workflow

- Dataset engineering & wrangling

- Visual storytelling (EDA)

- Machine learning & NLP

- Deployment with Streamlit

- Real-world recommendation system design

It is suitable to showcase in interviews, GitHub portfolios, and resume projects.

**🙌 8. Author**

***Kaushlendra Pratap Singh***
📍 New Delhi, India
📧 Available on request
💼 Seeking roles: Data Analyst, Senior Analyst, Data Scientist, Reporting Analyst


**⭐ If you like this project, consider giving the repository a star!**
