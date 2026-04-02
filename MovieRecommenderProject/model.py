import pandas
import sklearn
import scipy
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
DATASET_PATH = os.path.join(BASE_DIR, "movie_dataset.csv")


def load_movie_dataset(dataset_path: str = DATASET_PATH) -> pandas.DataFrame:
    data = pandas.read_csv(dataset_path)
    key_cols = [
        "title",
        "genres",
        "overview",
        "keywords",
        "runtime",
        "budget",
        "poster_path",
        "production_countries",
        "production_companies",
    ]
    data = data[key_cols]
    data = data[
        (data["budget"] > 100_000) &
        (data["runtime"] >= 60) &
        (data["runtime"] <= 300)
    ]
    data = data.reset_index(drop=True)
    return data


def preprocess_movie_data(data: pandas.DataFrame):
    genre_lists = data["genres"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )
    country_lists = data["production_countries"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )

    genre_encoder = sklearn.preprocessing.MultiLabelBinarizer()
    genres_data = genre_encoder.fit_transform(genre_lists)

    country_encoder = sklearn.preprocessing.MultiLabelBinarizer()
    country_data = country_encoder.fit_transform(country_lists)

    overview_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
    overview_data = overview_encoder.fit_transform(data["overview"].fillna(""))

    keyword_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
    keyword_data = keyword_encoder.fit_transform(data["keywords"].fillna(""))

    numeric_data = data[["runtime", "budget"]].copy()

    for col in ["runtime", "budget"]:
        median_val = numeric_data[col].median()
        numeric_data[col] = numeric_data[col].fillna(median_val)

    scaler = sklearn.preprocessing.MinMaxScaler()
    numeric_data_normalized = scaler.fit_transform(numeric_data)

    encoders = {
        "genres": genre_encoder,
        "overview": overview_encoder,
        "keywords": keyword_encoder,
        "production_countries": country_encoder,
        "numeric": scaler,
    }

    encoded_data = {
        "genres": genres_data,
        "overview": overview_data,
        "keywords": keyword_data,
        "production_countries": country_data,
        "numeric": numeric_data_normalized,
    }

    return encoders, encoded_data


def combine_movie_features(
    encoded_data: dict,
    weights: dict = {
        "genres": 2.0,
        "country": 1.1,
        "keyword": 1.5,
        "overview": 1.3,
        "numeric": 1.2,
    },
):
    weighted_genres = scipy.sparse.csr_matrix(
        encoded_data["genres"] * weights["genres"]
    )
    weighted_country = scipy.sparse.csr_matrix(
        encoded_data["production_countries"] * weights["country"]
    )
    weighted_keywords = encoded_data["keywords"] * weights["keyword"]
    weighted_overview = encoded_data["overview"] * weights["overview"]
    weighted_numeric = scipy.sparse.csr_matrix(
        encoded_data["numeric"] * weights["numeric"]
    )

    feature_data = scipy.sparse.hstack([
        weighted_genres,
        weighted_country,
        weighted_keywords,
        weighted_overview,
        weighted_numeric,
    ])

    return feature_data


def train_model(feature_data, n_neighbors: int = 30):
    model = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric="cosine",
    )
    model.fit(feature_data)
    return model


def recommend_movies(movie_title: str, data: pandas.DataFrame, feature_data, model, n_recommendations=20):
    match = data[data["title"].str.lower() == movie_title.lower()]

    if len(match) == 0:
        print("Try a new title")
        return

    movie_index = match.index[0]
    movie_vector = feature_data[movie_index]

    distances, indices = model.kneighbors(movie_vector)

    distances = distances.flatten()
    indices = indices.flatten()

    recommendations = []

    for distance, index in zip(
        distances[1:n_recommendations + 1],
        indices[1:n_recommendations + 1]
    ):
        rec_movie = data.loc[index]
        similarity = 1 - distance

        recommendations.append({
            "title": rec_movie["title"],
            "similarity": similarity,
            "genres": rec_movie["genres"],
            "budget": rec_movie["budget"],
            "keywords": rec_movie["keywords"],
            "overview": rec_movie["overview"],
            "countries": rec_movie["production_countries"],
            "poster_path": rec_movie["poster_path"],
        })

    return recommendations


def get_movie_info(movie_title: str, data: pandas.DataFrame) -> dict:
    match = data[data["title"].str.lower() == movie_title.lower()]
    if len(match) == 0:
        return None

    movie = match.iloc[0]
    return {
        "title": movie["title"],
        "genres": movie["genres"],
        "budget": movie["budget"],
        "overview": movie["overview"],
        "poster_path": movie["poster_path"],
    }


def save_model(model, feature_data, data, encoders, save_dir: str = MODEL_ARTIFACTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "knn_model.joblib"))
    joblib.dump(feature_data, os.path.join(save_dir, "feature_data.joblib"))
    data.to_csv(os.path.join(save_dir, "movie_data.csv"), index=False)
    joblib.dump(encoders, os.path.join(save_dir, "encoders.joblib"))

    print(f"All artifacts saved to '{save_dir}'")


def load_model(save_dir: str = MODEL_ARTIFACTS_DIR):
    model = joblib.load(os.path.join(save_dir, "knn_model.joblib"))
    feature_data = joblib.load(os.path.join(save_dir, "feature_data.joblib"))
    data = pandas.read_csv(os.path.join(save_dir, "movie_data.csv"))
    encoders = joblib.load(os.path.join(save_dir, "encoders.joblib"))

    print(f"Loaded model and {len(data):,} movies from '{save_dir}'")

    return model, feature_data, data, encoders


def train_and_save():
    data = load_movie_dataset(DATASET_PATH)
    encoders, encoded_data = preprocess_movie_data(data)
    feature_data = combine_movie_features(encoded_data)
    model = train_model(feature_data)
    save_model(model, feature_data, data, encoders, save_dir=MODEL_ARTIFACTS_DIR)
