import pandas as pd
import sklearn
import ast
import scipy
import keras
from rapidfuzz import process, fuzz
import pickle
import numpy
import os




def data_preprocess(csv_path:str="job-skills.csv"):
        
    df = pd.read_csv("job-skills.csv")

    job_description_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
    jobDescription = job_description_encoder.fit_transform(df['job_description'])

    job_title_encoder = sklearn.feature_extraction.text.TfidfVectorizer()
    jobTitle = job_title_encoder.fit_transform(df['job_title'])

    job_skill_set_encoder = sklearn.preprocessing.MultiLabelBinarizer()
    jobSkillset = job_skill_set_encoder.fit_transform(df['job_skill_set'].apply(ast.literal_eval))

    job_category_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
    jobCategory = job_category_encoder.fit_transform(df[['category']])

    x = scipy.sparse.hstack([jobDescription, jobSkillset, jobCategory, jobTitle])

    # Save the encoders
    encoders = {
        'description': job_description_encoder,
        'skills': job_skill_set_encoder,
        'category': job_category_encoder,
        'title' : job_title_encoder,
        
    }
    return df,x,encoders


def build_autoencoder(input_dim, embedding_dim=128):
    """
    Build an autoencoder model with the following architecture:
    
    Input (input_dim) 
        → Dense(512) → Dense(256) 
        → Dense(embedding_dim) [BOTTLENECK/EMBEDDING]
        → Dense(256) → Dense(512)
        → Output (input_dim)
    
    Args:
        input_dim: Number of input features
        embedding_dim: Size of the embedding layer (bottleneck)
    
    Returns:
        autoencoder: Full model for training
        encoder: Just the encoder part (for generating embeddings)
    """
    print("\n[2.1] Building autoencoder architecture...")
    
    # --- Encoder ---
    # Progressively compress the input down to the embedding
    inputs = keras.Input(shape=(input_dim,), name='input_layer')
    
    # First hidden layer: initial compression
    x = keras.layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-5),  # L2 regularization prevents overfitting
        name='encoder_dense_1'
    )(inputs)
    x = keras.layers.BatchNormalization(name='encoder_bn_1')(x)  # Stabilizes training
    x = keras.layers.Dropout(0.3, name='encoder_dropout_1')(x)   # Prevents overfitting
    
    # Second hidden layer: further compression
    x = keras.layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='encoder_dense_2'
    )(x)
    x = keras.layers.BatchNormalization(name='encoder_bn_2')(x)
    x = keras.layers.Dropout(0.2, name='encoder_dropout_2')(x)
    
    # Bottleneck layer: THE EMBEDDING
    # This is where the magic happens - the model must compress all important
    # information into just 128 dimensions
    embedding = keras.layers.Dense(
        embedding_dim, 
        activation='relu',
        name='embedding_layer'
    )(x)
    
    # --- Decoder ---
    # Mirror the encoder to reconstruct the original input
    x = keras.layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='decoder_dense_1'
    )(embedding)
    x = keras.layers.BatchNormalization(name='decoder_bn_1')(x)
    x = keras.layers.Dropout(0.2, name='decoder_dropout_1')(x)
    
    x = keras.layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name='decoder_dense_2'
    )(x)
    x = keras.layers.BatchNormalization(name='decoder_bn_2')(x)
    x = keras.layers.Dropout(0.3, name='decoder_dropout_2')(x)
    
    # Output layer: reconstruct the original input
    # Using sigmoid because our TF-IDF values are between 0 and 1
    outputs = keras.layers.Dense(
        input_dim, 
        activation='sigmoid',
        name='output_layer'
    )(x)
    
    # --- Create Models ---
    # Full autoencoder: for training (input → reconstructed input)
    autoencoder = keras.Model(inputs, outputs, name='autoencoder')
    
    # Encoder only: for inference (input → embedding)
    encoder = keras.Model(inputs, embedding, name='encoder')
    
    return autoencoder, encoder


def train_autoencoder(X, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train the autoencoder on the feature matrix.
    
    The model learns to reconstruct its input. A well-trained model means
    the embedding layer has captured the essential patterns in the data.
    
    Args:
        X: Sparse feature matrix
        epochs: Number of training epochs
        batch_size: Samples per gradient update
        validation_split: Fraction of data for validation
    
    Returns:
        autoencoder: Trained full model
        encoder: Trained encoder (for generating embeddings)
        history: Training history
    """
    X_dense = X.toarray().astype('float32')
    
    # Build the model
    autoencoder, encoder = build_autoencoder(X_dense.shape[1])
    
    print("\n[2.2] Compiling model...")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error: penalizes reconstruction errors
        metrics=['mae']  # Mean Absolute Error: easier to interpret
    )
    
    autoencoder.summary()
    
    # Set up callbacks for better training
    callbacks = [
        # Stop early if validation loss stops improving
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,           # Wait 10 epochs before stopping
            restore_best_weights=True  # Revert to best model
        ),
        # Reduce learning rate when loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,            # Halve the learning rate
            patience=5,            # Wait 5 epochs before reducing
            min_lr=1e-6
        )
    ]
    
    history = autoencoder.fit(
        X_dense, X_dense,  # Input and target are the same!
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, encoder, history


def generate_job_embeddings(encoder, X):
    """
    Generate embeddings for all jobs in the dataset.
    
    Args:
        encoder: Trained encoder model
        X: Sparse feature matrix
    
    Returns:
        embeddings: Dense matrix of job embeddings (n_jobs × embedding_dim)
    """
    print("\n" + "=" * 60)
    print("STEP 3: INFERENCE SETUP")
    print("=" * 60)
    
    print("\n[3.1] Generating embeddings for all jobs...")
    X_dense = X.toarray().astype('float32')
    embeddings = encoder.predict(X_dense, verbose=0)
    
    return embeddings


def save_model(encoder, encoders, df, job_embeddings):
    os.makedirs("save_dir", exist_ok=True)
    encoder.save(f"{"save_dir"}/encoder.keras")
    with open("save_dir/encoders.pkl","wb") as f:
        pickle.dump(encoders,f) 
    numpy.save("save_dir/job_embeddings.npy", job_embeddings)
    df.to_pickle("save_dir/jobs_df.pkl")
    print("save model")


# for idx, row in reccs.iterrows():
#     print(f"\n{row['job_title']}")
#     print(f"    Category: {row['category']}")
#     print(f"    Similarity: {row['similarity_score']:.4f}")
#     print(f"    Description: {row['job_description'][:100]}...")




