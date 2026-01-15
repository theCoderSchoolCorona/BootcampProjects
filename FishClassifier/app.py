"""
Fish Species Classifier - Streamlit App
========================================
A user-friendly interface for identifying fish species from images.
Uses a MobileNetV2-based transfer learning model trained on fish images.

This app follows the established layout pattern:
- Sidebar for image upload and settings
- Main area displays the image and prediction results
- Cached model loading for performance
"""

import streamlit as st
import numpy
import keras
import os
import plotly.express as px
import pandas as pd

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Must be the first Streamlit command - sets up the page layout and metadata

st.set_page_config(
    page_title="FinFinder",
    page_icon="🐟",
    layout="wide"
)

# =============================================================================
# CACHED MODEL LOADING
# =============================================================================
# Using st.cache_resource ensures the model loads only once and persists
# across all user interactions and page reruns - critical for performance
# since loading a Keras model can take several seconds

@st.cache_resource
def load_model():
    """
    Load the trained fish classification model.
    
    Cached to avoid reloading on every interaction - this is crucial for
    performance since model loading involves reading weights from disk
    and reconstructing the neural network architecture.
    
    Returns:
        model: The trained Keras model ready for inference
    """
    
    model_path = "./FishClassifier/fish_model.keras"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please ensure 'fish_model.keras' is in the same directory as this app.")
        st.stop()
    
    model = keras.models.load_model(model_path)
    return model


@st.cache_data
def get_class_names():
    """
    Get the list of fish species the model can identify.
    
    In a production app, this would typically be saved alongside the model.
    For now, we'll load it from the organized data directory structure.
    
    Returns:
        list: Sorted list of class names (fish species)
    """
    import pathlib
    
    organized_dir = pathlib.Path(".FishClassifier/Fish_Data/organized/")
    
    if organized_dir.exists():
        # Get class names from folder structure (same as how Keras loads them)
        class_names = sorted([d.name for d in organized_dir.iterdir() if d.is_dir()])
        return class_names
    else:
        st.warning("Could not find organized data directory. Using placeholder class names.")
        return ["Unknown"]


def predict_fish(image, model, class_names, show_n=5):
    """
    Run inference on an uploaded image.
    
    This function preprocesses the image to match the model's expected input
    format, runs the prediction, and returns the top N results.
    
    Args:
        image: PIL Image object from Streamlit's file_uploader
        model: The loaded Keras model
        class_names: List of species names corresponding to model output indices
        show_n: Number of top predictions to return
        
    Returns:
        predictions_list: List of tuples (species_name, confidence_percentage)
    """
    
    # Resize image to match model's expected input size (224x224)
    # The model was trained on this resolution using MobileNetV2
    image_resized = image.resize((224, 224))
    
    # Convert PIL image to numpy array
    image_array = keras.utils.img_to_array(image_resized)
    
    # Add batch dimension - model expects shape (batch_size, height, width, channels)
    # Even for single images, we need shape (1, 224, 224, 3)
    image_array = numpy.expand_dims(image_array, axis=0)
    
    # Run prediction - returns probabilities for each class
    predictions = model.predict(image_array, verbose=0)  # verbose=0 suppresses progress bar
    
    # Get indices of top N predictions, sorted by confidence (highest first)
    top_indices = numpy.argsort(predictions[0])[-show_n:][::-1]
    
    # Build list of (species_name, confidence_percentage) tuples
    predictions_list = [
        (class_names[idx], predictions[0][idx] * 100)
        for idx in top_indices
    ]
    
    return predictions_list


def format_species_name(name):
    """Convert species_name to Species Name for display."""
    return name.replace("_", " ").title()


# =============================================================================
# MAIN APP INTERFACE
# =============================================================================

# App title and description
st.title("FinFinder")
st.markdown("Upload an image of a fish to identify its species using Machine Learning!")

# -----------------------------------------------------------------------------
# SIDEBAR - User inputs and settings
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Upload Image")
    
    # File uploader for fish images
    uploaded_file = st.file_uploader(
        "Choose a fish image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a fish for best results"
    )
    
    st.divider()
    
    # Settings section
    st.header("Settings")
    
    # Number of predictions to show
    num_predictions = st.slider(
        "Number of predictions to show",
        min_value=1,
        max_value=10,
        value=5,
        help="How many top species predictions to display"
    )
    
    # Confidence threshold for "confident" prediction
    confidence_threshold = st.slider(
        "Confidence threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Predictions above this threshold are highlighted as confident"
    )
    
    st.divider()
    
    # About section
    st.header("About")
    st.markdown("""
    This classifier uses **EfficientNetB0** transfer learning
    to identify fish species from images.
    
    **How it works:**
    1. Upload a fish image
    2. The AI analyzes the image
    3. View the top species predictions
    
    The model was trained on a dataset of fish images
    organized by species.
    """)


# -----------------------------------------------------------------------------
# MAIN AREA - Results display
# -----------------------------------------------------------------------------

if uploaded_file is not None:
    # Load the image using PIL
    from PIL import Image
    image = Image.open(uploaded_file)
    
    # Load model and class names (cached, so only loads once)
    with st.spinner("Loading model..."):
        model = load_model()
        class_names = get_class_names()
    
    # Run prediction
    with st.spinner("Analyzing image..."):
        predictions = predict_fish(image, model, class_names, num_predictions)
    
    # Get top prediction info
    top_species, top_confidence = predictions[0]
    top_species_display = format_species_name(top_species)
    is_confident = top_confidence >= confidence_threshold
    
    # -------------------------------------------------------------------------
    # TOP PREDICTION BANNER
    # -------------------------------------------------------------------------
    
    st.subheader("Top Prediction")
    
    # Display metrics in columns
    metric_cols = st.columns([2, 1, 1])
    
    with metric_cols[0]:
        st.metric(
            label="Predicted Species",
            value=top_species_display
        )
    
    with metric_cols[1]:
        st.metric(
            label="Confidence",
            value=f"{top_confidence:.1f}%"
        )
    
    with metric_cols[2]:
        if is_confident:
            st.metric(label="Status", value="✓ Confident")
        else:
            st.metric(label="Status", value="? Uncertain")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # IMAGE AND CHART SIDE BY SIDE
    # -------------------------------------------------------------------------
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("All Predictions")
        
        # Build dataframe for Plotly chart
        chart_data = pd.DataFrame({
            "Species": [format_species_name(p[0]) for p in predictions],
            "Confidence": [p[1] for p in predictions],
            "Original": [p[0] for p in predictions]
        })
        
        # Reverse order so highest confidence is at top
        chart_data = chart_data.iloc[::-1]
        
        # Color bars based on confidence threshold
        chart_data["Color"] = chart_data["Confidence"].apply(
            lambda x: "Above Threshold" if x >= confidence_threshold else "Below Threshold"
        )
        
        # Create horizontal bar chart with Plotly
        fig = px.bar(
            chart_data,
            x="Confidence",
            y="Species",
            orientation="h",
            color="Color",
            color_discrete_map={
                "Above Threshold": "#2ecc71",  # Green
                "Below Threshold": "#3498db"   # Blue
            },
            text=chart_data["Confidence"].apply(lambda x: f"{x:.1f}%")
        )
        
        # Customize chart appearance
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Confidence (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 100]),
            margin=dict(l=0, r=20, t=10, b=40),
            font=dict(size=14)
        )
        
        # Position text outside bars for readability
        fig.update_traces(
            textposition="outside",
            textfont_size=12
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # No image uploaded yet - show instructions
    
    # Show example of what to expect
    st.markdown("### How to use this app:")
    st.markdown("""
    1. **Upload an image** using the file uploader in the sidebar
    2. **Adjust settings** if desired (number of predictions, confidence threshold)
    3. **View results** showing the predicted species and confidence levels
    """)
    
    # Show available species if we can load them
    class_names = get_class_names()
    if class_names and class_names[0] != "Unknown":
        with st.expander(f"Species this model can identify ({len(class_names)} total)"):
            # Display in columns for readability
            num_cols = 3
            cols = st.columns(num_cols)
            for i, species in enumerate(class_names):
                cols[i % num_cols].markdown(f"• {format_species_name(species)}")