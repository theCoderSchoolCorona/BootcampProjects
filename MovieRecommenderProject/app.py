import streamlit as st
import pandas as pd
import os
from model import train_and_save, load_model, recommend_movies, get_movie_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

KNN_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "knn_model.joblib")
FEATURE_DATA_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "feature_data.joblib")
MOVIE_DATA_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "movie_data.csv")
ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "encoders.joblib")

# TMDB image base URL, w342 is a good balance of quality and load speed
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# Placeholder for movies without posters
NO_POSTER_URL = "https://placehold.co/342x513/1a1a2e/ffffff?text=No+Poster"

st.set_page_config(
    page_title="GoWatch",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    """Load the trained model and data. Cached to avoid reloading on every interaction."""
    required_files = [
        KNN_MODEL_PATH,
        FEATURE_DATA_PATH,
        MOVIE_DATA_PATH,
        ENCODERS_PATH
    ]

    if not all(os.path.exists(path) for path in required_files):
        train_and_save()

    model, feature_data, data, encoders = load_model(MODEL_ARTIFACTS_DIR)
    return model, feature_data, data

# Load on startup
model, feature_data, data = load_recommender()

def get_poster_url(poster_path):
    if pd.notna(poster_path) and str(poster_path).strip():
        return TMDB_IMAGE_BASE + str(poster_path)
    return NO_POSTER_URL

def format_budget(budget):
    """Format budget as readable string."""
    if budget >= 1_000_000:
        return f"${budget / 1_000_000:.0f}M"
    elif budget >= 1_000:
        return f"${budget / 1_000:.0f}K"
    return f"${budget:,.0f}"

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("GoWatch: A Movie Recommender")
st.write("Find similar movies based on genres, plot, keywords, and more!")

st.sidebar.header("Settings")

movie_list = sorted(data["title"].tolist())
selected_movie = st.sidebar.selectbox(
    "Choose a movie you like:",
    options=movie_list,
    index=movie_list.index("Inception") if "Inception" in movie_list else 0
)

n_recommendations = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=15,
    value=6
)

st.sidebar.subheader("Filter by Budget")
budget_filter = st.sidebar.radio(
    "Show only:",
    options=["All Movies", "Indie (<\\$30M)", "Mid (\\$30M-\\$100M)", "Blockbuster (>\\$100M)"]
)

# ==============================================================================
# MAIN CONTENT
# ==============================================================================

st.subheader("Your Selection")
input_movie = get_movie_info(selected_movie, data)

if input_movie:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(get_poster_url(input_movie["poster_path"]), width=200)

    with col2:
        st.markdown(f"### {input_movie['title']}")
        st.write(f"**Genres:** {input_movie['genres']}")
        st.write(f"**Budget:** {format_budget(input_movie['budget'])}")
        st.write(f"**Overview:** {input_movie['overview']}")

st.divider()

st.subheader(f"Movies Similar to '{selected_movie}'")

recommendations = recommend_movies(
    movie_title=selected_movie,
    data=data,
    feature_data=feature_data,
    model=model,
    n_recommendations=n_recommendations + 10
)

if recommendations:
    if budget_filter == "Indie (<\\$30M)":
        recommendations = [r for r in recommendations if r["budget"] < 30_000_000]
    elif budget_filter == "Mid (\\$30M-\\$100M)":
        recommendations = [r for r in recommendations if 30_000_000 <= r["budget"] < 100_000_000]
    elif budget_filter == "Blockbuster (>\\$100M)":
        recommendations = [r for r in recommendations if r["budget"] >= 100_000_000]

    recommendations = recommendations[:n_recommendations]

    if not recommendations:
        st.warning("No movies found matching your budget filter. Try a different filter!")
    else:
        cols = st.columns(3)

        for i, movie in enumerate(recommendations):
            with cols[i % 3]:
                st.image(get_poster_url(movie["poster_path"]), use_container_width=True)
                st.markdown(f"**{movie['title']}**")
                st.progress(movie["similarity"], text=f"{movie['similarity']:.0%} match")

                with st.expander("Details"):
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Budget:** {format_budget(movie['budget'])}")
                    st.write(f"**Countries:** {movie['countries']}")
                    st.write(f"**Overview:** {movie['overview']}")
else:
    st.error("Movie not found! Please try a different title.")

st.divider()
st.caption(f"Database: {len(data):,} movies | Model: K-Nearest Neighbors with TF-IDF features")
