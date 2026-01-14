import streamlit as st
import pandas as pd
import os
import re

# Import our model and inference functions
from model import data_preprocess, train_autoencoder, generate_job_embeddings, save_model
from inf import load_model, recommend_jobs

if 'expanded_jobs' not in st.session_state:
    st.session_state.expanded_jobs = set()
    
# =============================================================================
# CACHED MODEL LOADING
# =============================================================================
# Using st.cache_resource ensures the model loads only once and persists
# across all user interactions and page reruns

@st.cache_resource
def load_recommender():
    """
    Load or train the job recommendation model.
    
    Cached to avoid reloading on every interaction - this is crucial for
    performance since loading/training can take significant time.
    
    Returns:
        encoder: The trained encoder model (bottleneck of autoencoder)
        encoders: Dictionary of fitted preprocessors (TF-IDF, OneHot, etc.)
        job_embeddings: Pre-computed embeddings for all jobs
        df: DataFrame containing all job listings
    """
    if not os.path.exists("save_dir"):
        # First run: preprocess data and train model
        st.info("First run detected. Training model... This may take a few minutes.")
        df, x, encoders = data_preprocess()
        autoencoder, encoder, history = train_autoencoder(
            x, epochs=50, batch_size=32, validation_split=0.2
        )
        job_embeddings = generate_job_embeddings(encoder, x)
        save_model(encoder, encoders, df, job_embeddings)
    else:
        # Load pre-trained model
        encoder, encoders, job_embeddings, df = load_model()
    
    return encoder, encoders, job_embeddings, df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def get_categories(_df):
    """Extract unique job categories from the dataframe."""
    return sorted(_df['category'].unique().tolist())


@st.cache_data  
def get_common_skills(_encoders):
    """
    Get list of skills the model knows about.
    These come from the MultiLabelBinarizer fitted during training.
    """
    return sorted(_encoders['skills'].classes_.tolist())

def truncate_text(text, max_length=300):
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def capitalize_sentences(text):
    """Capitalize the first letter of each sentence."""
    if not text:
        return text
    
    # Capitalize first character
    result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Capitalize letter after sentence-ending punctuation followed by space
    result = re.sub(
        r'([.!?]\s*)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        result
    )
    
    return result
# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="JobMatch",
    page_icon="💼",
    layout="wide"
)

# Load the model (cached - only runs once)
encoder, encoders, job_embeddings, df = load_recommender()

# Get available categories and skills for dropdowns
categories = get_categories(df)
known_skills = get_common_skills(encoders)


# =============================================================================
# HEADER
# =============================================================================

st.title("AI-Powered Job Recommender")
st.write("Find jobs that match your skills, experience, and preferences using deep learning embeddings.")


# =============================================================================
# SIDEBAR - User Inputs
# =============================================================================

# st.sidebar.header("Your Profile")

# Category selection - required for filtering
selected_category = st.sidebar.selectbox(
    "Job Category",
    options=categories,
    help="Select the job category you're interested in"
)

# Job title input - what kind of role are you looking for?
user_title = st.sidebar.text_input(
    "Desired Job Title",
    placeholder="e.g., Data Scientist, Software Engineer",
    help="Enter the type of role you're looking for"
)

# Skills input - using multiselect for known skills + text input for others
# st.sidebar.subheader("Your Skills")

# Multiselect for skills the model knows about
selected_skills = st.sidebar.multiselect(
    "Select from known skills:",
    options=known_skills,
    help="Select skills you have. The model uses fuzzy matching, so close matches work too!"
)

# Text input for additional skills not in the list
additional_skills = st.sidebar.text_input(
    "Additional skills (comma-separated):",
    placeholder="e.g., TensorFlow, React, Agile",
    help="Add any skills not in the dropdown above"
)

# Combine selected and additional skills
all_skills = selected_skills.copy()
if additional_skills:
    # Parse comma-separated skills and clean them up
    extra = [s.strip() for s in additional_skills.split(',') if s.strip()]
    all_skills.extend(extra)

# Show selected skills count
if all_skills:
    st.sidebar.caption(f" {len(all_skills)} skills selected")

# Job description - what are you looking for?
# st.sidebar.subheader("Describe Your Ideal Job")
user_description = st.sidebar.text_area(
    "What kind of work are you looking for?",
    placeholder="Describe your ideal job, responsibilities you want, technologies you want to work with...",
    height=120,
    help="Be specific! The more detail you provide, the better the recommendations."
)

# Number of recommendations
st.sidebar.divider()
n_recommendations = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=15,
    value=5
)


# =============================================================================
# MAIN CONTENT - Recommendations
# =============================================================================

st.divider()

# Check if user has provided enough input
if not user_title and not user_description and not all_skills:
    # Show some stats about the dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jobs", f"{len(df):,}")
    with col2:
        st.metric("Categories", len(categories))
    with col3:
        st.metric("Unique Skills", len(known_skills))
    
    # Show sample jobs from selected category
    st.subheader(f"Sample Jobs in '{selected_category}'")
    sample_jobs = df[df['category'] == selected_category].head(6)
    
    cols = st.columns(3)
    for i, (_, job) in enumerate(sample_jobs.iterrows()):
        with cols[i % 3]:
            st.markdown(f"**{job['job_title']}**")
            st.caption(job['job_description'])

else:
    # User has provided input - show recommendations
    st.subheader(f"Recommended Jobs for You")
    
    # Provide defaults if user left fields empty
    search_description = user_description if user_description else user_title
    search_title = user_title if user_title else "any"
    search_skills = all_skills if all_skills else []
    
    if not search_description and not search_skills:
        st.warning("Please provide at least a job description OR some skills to get recommendations.")
    else:
        # Call the recommendation function
        with st.spinner("Finding matching jobs..."):
            recommendations = recommend_jobs(
                user_description=search_description,
                user_skills=search_skills,
                user_category=selected_category,
                user_title=search_title,
                encoder=encoder,
                encoders=encoders,
                job_embeddings=job_embeddings,
                df=df,
                top_n=n_recommendations
            )
        
        if recommendations.empty:
            st.warning(f"No jobs found in category '{selected_category}'. Try a different category!")
        else:
            # Display recommendations in a grid with balanced cards
            
            # Process jobs into pairs for balanced rows
            jobs_list = list(recommendations.iterrows())
            
            for row_start in range(0, len(jobs_list), 2):
                # Get jobs for this row (1 or 2 jobs)
                row_jobs = jobs_list[row_start:row_start + 2]
                cols = st.columns(2)
                
                for col_idx, (idx, job) in enumerate(row_jobs):
                    with cols[col_idx]:
                        # Use container with border for card-like appearance
                        with st.container(border=True):
                            # Job title
                            st.markdown(f"**{job['job_title']}**")
                            
                            # Similarity as progress bar
                            similarity_pct = job['similarity_score']
                            st.progress(
                                min(similarity_pct, 1.0),
                                text=f"{similarity_pct:.0%} match"
                            )
                            
                            # Category badge
                            st.caption(f"{job['category']}")
                            
                            # Job description with toggle
                            job_key = f"job_{idx}"
                            
                            with st.expander("View Description"):
                                if job_key in st.session_state.expanded_jobs:
                                    st.write(capitalize_sentences(job['job_description'].lower()))
                                    if st.button("Show Less", key=f"less_{job_key}"):
                                        st.session_state.expanded_jobs.discard(job_key)
                                        st.rerun()
                                else:
                                    st.write(capitalize_sentences(job['job_description'][:150].lower()))
                                    if st.button("Show More", key=f"more_{job_key}"):
                                        st.session_state.expanded_jobs.add(job_key)
                                        st.rerun()


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

# Footer with dataset info
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.caption(
        f"Database: {len(df):,} jobs | "
        f"{len(categories)} categories | "
        f"{len(known_skills):,} unique skills"
    )
with footer_col2:
    st.caption(
        "Model: Autoencoder embeddings with cosine similarity | "
        "Built with Streamlit"
    )