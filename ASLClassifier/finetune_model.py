
# python finetune_model.py --samples collected_samples --epochs 10
    

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import keras
from keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# =============================================================================
# CONFIGURATION
# =============================================================================

ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

LETTER_TO_LABEL = {letter: i for i, letter in enumerate(ASL_LETTERS)}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_custom_samples(samples_dir):
    """
    Load samples collected with asl_collect_data.py.
    
    Returns arrays in the same format as Sign MNIST:
    - X: (N, 28, 28, 1) normalized float32
    - y: (N,) integer labels
    """
    import cv2
    
    samples_dir = Path(samples_dir)
    
    images = []
    labels = []
    
    for letter in ASL_LETTERS:
        letter_dir = samples_dir / letter
        if not letter_dir.exists():
            continue
        
        label = LETTER_TO_LABEL[letter]
        
        for img_path in letter_dir.glob("*.png"):
            # Load as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Ensure correct size
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28))
            
            images.append(img)
            labels.append(label)
    
    if not images:
        return None, None
    
    # Convert to numpy arrays
    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int32')
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    # Add channel dimension
    X = X.reshape(-1, 28, 28, 1)
    
    print(f"Loaded {len(images)} custom samples")
    
    # Show distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Samples per letter:")
    for label, count in zip(unique, counts):
        letter = ASL_LETTERS[label]
        print(f"  {letter}: {count}")
    
    return X, y


def load_sign_mnist(train_path, test_path=None, sample_fraction=0.1):
    """
    Load a fraction of Sign MNIST data to mix with custom samples.
    This helps prevent catastrophic forgetting during fine-tuning.
    """
    train_df = pd.read_csv(train_path)
    
    # Sample a fraction of the data
    if sample_fraction < 1.0:
        train_df = train_df.sample(frac=sample_fraction, random_state=42)
    
    y = train_df['label'].values
    X = train_df.drop('label', axis=1).values
    
    # Normalize and reshape
    X = X.astype('float32') / 255.0
    X = X.reshape(-1, 28, 28, 1)
    
    print(f"Loaded {len(X)} Sign MNIST samples ({sample_fraction:.0%} of dataset)")
    
    return X, y


def prepare_training_data(custom_X, custom_y, mnist_X=None, mnist_y=None, 
                          custom_weight=3.0, val_split=0.2):
    """
    Combine custom samples with Sign MNIST and prepare for training.
    
    Custom samples are weighted higher to emphasize learning from camera data
    while Sign MNIST prevents forgetting.
    """
    # Combine datasets
    if mnist_X is not None:
        X_combined = np.concatenate([custom_X, mnist_X])
        y_combined = np.concatenate([custom_y, mnist_y])
        
        # Create sample weights: custom samples get higher weight
        weights = np.ones(len(X_combined))
        weights[:len(custom_X)] = custom_weight
    else:
        X_combined = custom_X
        y_combined = custom_y
        weights = np.ones(len(X_combined))
    
    # Shuffle
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    weights = weights[indices]
    
    # Split into train/val
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_combined, y_combined, weights,
        test_size=val_split,
        stratify=y_combined,
        random_state=42
    )
    
    # One-hot encode labels
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(24))  # 24 classes
    y_train = label_binarizer.transform(y_train)
    y_val = label_binarizer.transform(y_val)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    return X_train, y_train, X_val, y_val, w_train


def prepare_model_for_finetuning(model_path, freeze_layers=6, learning_rate=0.0001):
    """
    Load pre-trained model and prepare it for fine-tuning.
    
    Strategy:
    1. Freeze early convolutional layers (they learn edges, textures - universal)
    2. Keep later layers trainable (they learn hand-specific features)
    3. Use a very low learning rate to preserve learned features
    """
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Print layer info
    print("\nModel layers:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
    
    # Freeze early layers
    # Typically freeze the first few conv blocks, keep later ones trainable
    print(f"\nFreezing first {freeze_layers} layers...")
    
    for i, layer in enumerate(model.layers):
        if i < freeze_layers:
            layer.trainable = False
            print(f"  Frozen: {layer.name}")
        else:
            layer.trainable = True
            print(f"  Trainable: {layer.name}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    return model



def finetune_model(model, X_train, y_train, X_val, y_val, sample_weights,
                   epochs=10, batch_size=32):
    """
    Fine-tune the model with collected samples.
    """
    # Callbacks
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'asl_model_finetuned_{timestamp}.keras'
    
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        sample_weight=sample_weights,
        callbacks=[reduce_lr, early_stop, checkpoint],
        verbose=1
    )
    
    return history, checkpoint_path


def plot_finetuning_history(history, save_path='finetuning_history.png'):
    """Plot fine-tuning progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Fine-tuning Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Fine-tuning Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Saved training history plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ASL model with custom samples')
    
    parser.add_argument('--model', type=str, default='main_model.keras',
                       help='Path to pre-trained model')
    parser.add_argument('--samples', type=str, default='collected_samples',
                       help='Directory containing collected samples')
    parser.add_argument('--mnist-train', type=str, default=None,
                       help='Path to Sign MNIST training CSV (optional)')
    parser.add_argument('--mnist-fraction', type=float, default=0.1,
                       help='Fraction of Sign MNIST to include')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--freeze-layers', type=int, default=6,
                       help='Number of layers to freeze')
    parser.add_argument('--custom-weight', type=float, default=3.0,
                       help='Weight multiplier for custom samples')
    
    args = parser.parse_args()
    
    
    # Check prerequisites
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.samples):
        print(f"Error: Samples directory not found at {args.samples}")
        print("Run asl_collect_data.py first to collect training samples.")
        return
    
    # Load custom samples
    print("Loading custom samples...")
    
    custom_X, custom_y = load_custom_samples(args.samples)
    
    if custom_X is None or len(custom_X) < 10:
        print("Error: Not enough custom samples. Collect at least 10 samples.")
        return
    
    # Optionally load Sign MNIST
    mnist_X, mnist_y = None, None
    if args.mnist_train and os.path.exists(args.mnist_train):
        print("Loading Sign MNIST samples...")
        mnist_X, mnist_y = load_sign_mnist(args.mnist_train, 
                                           sample_fraction=args.mnist_fraction)
    
    # Prepare training data
    
    X_train, y_train, X_val, y_val, weights = prepare_training_data(
        custom_X, custom_y,
        mnist_X, mnist_y,
        custom_weight=args.custom_weight
    )
    
    # Prepare model
    
    model = prepare_model_for_finetuning(
        args.model,
        freeze_layers=args.freeze_layers,
        learning_rate=args.learning_rate
    )
    
    # Fine-tune
    
    history, checkpoint_path = finetune_model(
        model, X_train, y_train, X_val, y_val, weights,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot results
    
    plot_finetuning_history(history)
    
    # Final evaluation
    
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")
    
    # Save final model
    final_path = 'asl_model_finetuned.keras'
    model.save(final_path)
    print(f"  MODEL_PATH = '{final_path}'")


if __name__ == "__main__":
    main()