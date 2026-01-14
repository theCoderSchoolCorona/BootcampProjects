"""
ASL (American Sign Language) Classification - Training Script
==============================================================
This script trains a CNN to recognize ASL letters from the Sign MNIST dataset.
The dataset contains 28x28 grayscale images of hand signs for letters A-Y (excluding J).

Updates from original:
- Modern Keras 3 API (compatible with TensorFlow 2.16+)
- Enhanced regularization to address overfitting
- Early stopping and model checkpointing
- Native .keras format instead of legacy .h5
- Built-in data augmentation layers instead of deprecated ImageDataGenerator
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging noise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Use keras directly (works with both standalone Keras 3 and tf.keras)
import keras
from keras import layers, models, callbacks, regularizers

print(f"Keras version: {keras.__version__}")

# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(train_path, test_path):
    """
    Load the Sign MNIST dataset and prepare it for training.
    
    The dataset uses labels 0-24, but skips 9 (letter J) because J requires motion.
    This gives us 24 classes total for static hand signs.
    """
    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate labels from pixel data
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    x_train = train_df.drop('label', axis=1).values
    x_test = test_df.drop('label', axis=1).values
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to (samples, height, width, channels) for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)  # Use transform, not fit_transform!
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    return x_train, y_train, x_test, y_test, label_binarizer


# =============================================================================
# STEP 2: MODEL ARCHITECTURE
# =============================================================================

def create_model(input_shape=(28, 28, 1), num_classes=24):
    """
    Create an improved CNN with better regularization to prevent overfitting.
    
    Key improvements over original:
    - Built-in data augmentation layers (replaces deprecated ImageDataGenerator)
    - L2 regularization on convolutional layers
    - More aggressive dropout
    - Smaller dense layer to reduce parameter count
    """
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Data augmentation (only active during training)
        layers.RandomRotation(0.05),           # ±18 degrees
        layers.RandomZoom(0.1),                # ±10% zoom
        layers.RandomTranslation(0.1, 0.1),    # ±10% shift
        
        # First convolutional block
        layers.Conv2D(
            64, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(
            128, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(
            64, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.5),  # Higher dropout before output
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# STEP 3: TRAINING WITH CALLBACKS
# =============================================================================

def train_model(model, x_train, y_train, x_test, y_test, epochs=30, batch_size=128):
    """
    Train the model with callbacks to prevent overfitting and save best weights.
    """
    
    # Callback: Reduce learning rate when validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Callback: Stop training early if validation loss stops improving
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    # Callback: Save the best model during training
    checkpoint = callbacks.ModelCheckpoint(
        'asl_model_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop, checkpoint],
        verbose=1
    )
    
    return history


# =============================================================================
# STEP 4: EVALUATION AND VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """Plot training and validation metrics to visualize overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and print detailed metrics."""
    
    # ASL letters (J and Z excluded - they require motion)
    letter_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                     'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=letter_labels))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=letter_labels,
        yticklabels=letter_labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # File paths - adjust these to match your dataset location
    TRAIN_PATH = "dataset/sign_mnist_train.csv"
    TEST_PATH = "dataset/sign_mnist_test.csv"
    
    # Load and preprocess data
    print("Loading dataset...")
    x_train, y_train, x_test, y_test, label_binarizer = load_and_preprocess_data(
        TRAIN_PATH, TEST_PATH
    )
    
    # Create and display model
    print("\nBuilding model...")
    model = create_model()
    model.summary()
    
    # Train the model
    print("\nStarting training...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, x_test, y_test)
    
    # Save final model
    model.save('asl_model_final.keras')
    print("\nModel saved as 'asl_model_final.keras'")
    print("Best model saved as 'asl_model_best.keras'")