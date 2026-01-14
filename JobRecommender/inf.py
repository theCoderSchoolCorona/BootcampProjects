import pandas as pd
import sklearn
import scipy
import keras
from rapidfuzz import process, fuzz
import pickle
import numpy


def load_model():
    encoder=keras.models.load_model("save_dir/encoder.keras")
    with open("save_dir/encoders.pkl","rb") as f:
        encoders=pickle.load(f)
    job_embeddings=numpy.load("save_dir/job_embeddings.npy")
    df =pd.read_pickle("save_dir/jobs_df.pkl")
    print("loaded")
    return encoder, encoders, job_embeddings, df

def encode_user_sills_fuzzy(user_skills, skills_encoder, threshold=80):
    print(user_skills)
    known_skills = list(skills_encoder.classes_)
    matched_skills = []
    for skill in user_skills:
        if skill in known_skills:
            matched_skills.append(skill)
        else:
            match, score, _ = process.extractOne(
                skill, known_skills, scorer=fuzz.WRatio
            )
            if score >= threshold:
                matched_skills.append(match)
            else:
                print(f"No good matched skill for {skill}")
    print(matched_skills)
    return skills_encoder.transform([matched_skills])

def encode_user_input(user_description, user_skills, user_category, user_title, encoders):
    """
    Encode a user's job preferences into the same feature space as our training data.
    
    This is crucial: we must transform user input EXACTLY the same way we
    transformed the training data, using the SAME fitted encoders.
    
    Args:
        user_description: Text describing desired job (string)
        user_skills: List of skills the user has (list of strings)
        user_category: Preferred job category (string)
        encoders: Dictionary of fitted encoders from preprocessing
    
    Returns:
        X_user: Encoded feature vector for the user (sparse matrix)
    """
    # Encode description using the SAME TF-IDF vectorizer
    # transform() uses vocabulary learned during fit()
    X_desc = encoders['description'].transform([user_description])
    
    # Encode skills using the SAME MultiLabelBinarizer
    # Unknown skills are simply ignored (not in the learned vocabulary)
    X_skills = encode_user_sills_fuzzy(user_skills, encoders["skills"])
    
    # Encode category using the SAME OneHotEncoder
    # Must be a DataFrame with the same column name
    cat_df=pd.DataFrame({'category':[user_category]})
    X_cat = encoders['category'].transform(cat_df)

    
    x_title = encoders['title'].transform([user_title])
    
    # Combine in the SAME order as training
    X_user = scipy.sparse.hstack([X_desc, X_skills, X_cat, x_title])
    
    return X_user


def recommend_jobs(user_description, user_skills, user_category, user_title, 
                   encoder, encoders, job_embeddings, df, top_n=5):
    """
    Recommend jobs based on user input, with strict category matching.
    
    Process:
    1. Filter to only jobs in the requested category
    2. Encode user input into feature vector
    3. Pass through encoder to get user's embedding
    4. Compute cosine similarity with filtered job embeddings
    5. Return top N most similar jobs
    
    Args:
        user_description: What kind of job the user wants
        user_skills: Skills the user has
        user_category: Preferred category
        encoder: Trained encoder model
        encoders: Dictionary of fitted encoders
        job_embeddings: Pre-computed embeddings for all jobs
        df: Original dataframe with job details
        top_n: Number of recommendations to return
    
    Returns:
        recommendations: DataFrame with top job matches and similarity scores
    """
    
    # Step 1: Filter to only jobs in the requested category
    # This ensures all recommendations match the user's category selection
    category_mask = df['category'] == user_category
    
    # Use numpy.where to get POSITIONAL indices (0-based row numbers)
    # This avoids issues with non-standard pandas index labels
    filtered_positions = numpy.where(category_mask)[0]
    filtered_embeddings = job_embeddings[filtered_positions]
    
    print(f"Filtered to {len(filtered_positions)} jobs in category: {user_category}")
    
    # Handle edge case: no jobs in this category
    if len(filtered_positions) == 0:
        print(f"Warning: No jobs found in category '{user_category}'")
        return pd.DataFrame(columns=['job_title', 'category', 'job_description', 'similarity_score'])
    
    # Step 2: Encode user input
    X_user = encode_user_input(user_description, user_skills, user_category, user_title, encoders)
    
    # Step 3: Generate user embedding
    X_user_dense = X_user.toarray().astype('float32')
    user_embedding = encoder.predict(X_user_dense, verbose=0)
    
    # Step 4: Compute similarity with FILTERED jobs only
    # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
    similarities = sklearn.metrics.pairwise.cosine_similarity(user_embedding, filtered_embeddings)[0]
    
    # Step 5: Get top N indices (within the filtered set)
    # Limit top_n to the number of available jobs in this category
    actual_top_n = min(top_n, len(filtered_positions))
    top_local_indices = similarities.argsort()[::-1][:actual_top_n]
    
    # Map back to original dataframe positions using numpy indexing
    top_original_positions = filtered_positions[top_local_indices]
    
    # Build recommendations DataFrame using iloc (positional indexing)
    recommendations = df.iloc[top_original_positions][['job_title', 'category', 'job_description']].copy()
    recommendations['similarity_score'] = similarities[top_local_indices]
    recommendations['job_description'] = recommendations['job_description'].str[:200] + '...'
    
    return recommendations