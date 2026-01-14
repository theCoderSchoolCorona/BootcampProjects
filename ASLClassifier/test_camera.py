"""
ASL (American Sign Language) Real-Time Recognition
===================================================
Uses the modern MediaPipe Tasks API (compatible with TensorFlow 2.16+)
for hand detection and a trained CNN model for letter prediction.

Usage:
    python test_camera.py
    
Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
    - Press 'd' to toggle debug view

Requirements:
    pip install tensorflow opencv-python mediapipe
    
Note: On first run, the script will automatically download the hand landmarker
model file (~10MB) from Google's servers.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import keras
import urllib.request
from pathlib import Path
from collections import deque

# =============================================================================
# MEDIAPIPE TASKS API IMPORTS
# =============================================================================
# The new API lives under mediapipe.tasks instead of mp.solutions
# Note: The legacy mp.solutions module has been removed in newer MediaPipe versions
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = 'asl_model_finetuned.keras'

# Path where we'll store the MediaPipe hand landmarker model
# This gets downloaded automatically on first run
HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'
HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Detection thresholds
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
PREDICTION_CONFIDENCE_THRESHOLD = 0.5

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ASL letters (J and Z excluded - they require motion)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]


# =============================================================================
# MODEL DOWNLOAD HELPER
# =============================================================================

def ensure_hand_landmarker_model():
    """
    Download the MediaPipe hand landmarker model if it doesn't exist.
    
    The new Tasks API requires a .task model file rather than loading
    weights internally like the old solutions API did.
    """
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print(f"Downloading hand landmarker model...")
        print(f"  From: {HAND_LANDMARKER_MODEL_URL}")
        print(f"  To: {HAND_LANDMARKER_MODEL_PATH}")
        
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
            print("  Download complete!")
        except Exception as e:
            print(f"  Error downloading model: {e}")
            print("  Please download manually from the URL above.")
            return False
    
    return True


# =============================================================================
# HAND PREPROCESSOR
# =============================================================================

class HandPreprocessor:
    """
    Preprocesses hand images to bridge the domain gap between
    camera input and Sign MNIST training data.
    """
    
    def __init__(self, target_size=28):
        self.target_size = target_size
        
        # Skin detection thresholds in YCrCb color space
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)
    
    def create_hand_mask(self, frame, hand_landmarks, frame_width, frame_height):
        """
        Create a binary mask isolating the hand from background.
        
        The new API returns normalized landmarks (0-1 range), so we need
        to scale them to pixel coordinates using frame dimensions.
        """
        h, w = frame_height, frame_width
        
        # Extract landmark points and convert to pixel coordinates
        # In the new API, landmarks are NormalizedLandmark objects with x, y, z attributes
        landmark_points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            landmark_points.append([px, py])
        
        landmark_points = np.array(landmark_points, dtype=np.int32)
        hull = cv2.convexHull(landmark_points)
        
        # Create hull mask
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        
        # Dilate to capture full fingers
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        hull_mask = cv2.dilate(hull_mask, kernel, iterations=1)
        
        # Skin color detection for edge refinement
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_small)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(hull_mask, skin_mask)
        
        # Fall back to hull only if skin detection fails
        if cv2.countNonZero(combined_mask) < 1000:
            combined_mask = hull_mask
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask, landmark_points
    
    def get_centered_square_crop(self, mask, frame_gray, padding_ratio=0.15):
        """Extract a square crop centered on the hand's center of mass."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        moments = cv2.moments(largest_contour)
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Make square using larger dimension
        size = int(max(w, h) * (1 + padding_ratio * 2))
        half_size = size // 2
        
        frame_h, frame_w = frame_gray.shape
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(frame_w, cx + half_size)
        y2 = min(frame_h, cy + half_size)
        
        # Adjust if we hit frame boundaries
        if x2 - x1 < size and x1 == 0:
            x2 = min(frame_w, size)
        if x2 - x1 < size and x2 == frame_w:
            x1 = max(0, frame_w - size)
        if y2 - y1 < size and y1 == 0:
            y2 = min(frame_h, size)
        if y2 - y1 < size and y2 == frame_h:
            y1 = max(0, frame_h - size)
        
        crop_gray = frame_gray[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        
        return crop_gray, crop_mask, (x1, y1, x2, y2)
    
    def process(self, frame, hand_landmarks, frame_width, frame_height):
        """
        Main preprocessing pipeline.
        
        Returns:
            model_input: Tensor ready for model prediction (1, 28, 28, 1)
            debug_image: Visualization of what model sees (28, 28) uint8
            bbox: Bounding box coordinates for drawing
        """
        # Create hand mask
        mask, landmark_points = self.create_hand_mask(
            frame, hand_landmarks, frame_width, frame_height
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get centered square crop
        crop_result = self.get_centered_square_crop(mask, gray)
        if crop_result is None:
            return None, None, None
        
        hand_crop, mask_crop, bbox = crop_result
        
        if hand_crop.size == 0:
            return None, None, None
        
        # Apply mask (white background)
        masked_hand = hand_crop.copy()
        masked_hand[mask_crop == 0] = 255
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked_hand)
        
        # Resize to model input size
        resized = cv2.resize(enhanced, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_AREA)
        
        # Slight blur to match Sign MNIST smoothness
        smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Normalize to [0, 1]
        normalized = smoothed.astype('float32') / 255.0
        
        # Reshape for model
        model_input = normalized.reshape(1, self.target_size, self.target_size, 1)
        
        # Debug image for visualization
        debug_image = smoothed
        
        return model_input, debug_image, landmark_points


# =============================================================================
# ASL RECOGNIZER (using new MediaPipe Tasks API)
# =============================================================================

class ASLRecognizer:
    """
    Main recognition class using the modern MediaPipe Tasks API.
    
    Key differences from the old mp.solutions API:
    1. We create a HandLandmarker object with explicit options
    2. Results come as HandLandmarkerResult with hand_landmarks list
    3. Each hand's landmarks are a list of NormalizedLandmark objects
    4. We need to convert MP Image format for processing
    """
    
    def __init__(self, model_path):
        # Load ASL classification model
        print(f"Loading ASL model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("ASL model loaded successfully!")
        
        # Initialize the new MediaPipe Hand Landmarker
        print("Initializing MediaPipe Hand Landmarker...")
        
        # Create options for the hand landmarker
        # The new API uses an options pattern for configuration
        base_options = mp_tasks.BaseOptions(
            model_asset_path=HAND_LANDMARKER_MODEL_PATH
        )
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # Process single images
            num_hands=1,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("Hand Landmarker initialized!")
        
        # Initialize preprocessor
        self.preprocessor = HandPreprocessor()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=7)
        
        # Debug mode
        self.show_debug = True
        self.last_debug_image = None
    
    def predict_letter(self, model_input):
        """Get prediction from ASL model."""
        prediction = self.model.predict(model_input, verbose=0)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)
        
        if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
            letter = ASL_LETTERS[predicted_index]
        else:
            letter = "?"
        
        return letter, confidence, prediction[0]
    
    def smooth_prediction(self, letter, confidence):
        """Smooth predictions to reduce flickering."""
        self.prediction_history.append((letter, confidence))
        
        letter_scores = {}
        for i, (hist_letter, hist_conf) in enumerate(self.prediction_history):
            if hist_letter != "?":
                recency_weight = (i + 1) / len(self.prediction_history)
                score = hist_conf * recency_weight
                letter_scores[hist_letter] = letter_scores.get(hist_letter, 0) + score
        
        if letter_scores:
            return max(letter_scores, key=letter_scores.get)
        return "?"
    
    def draw_landmarks_manual(self, frame, landmark_points):
        """
        Draw hand landmarks manually since the new API doesn't return
        the same landmark format as the old drawing utils expect.
        """
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start = tuple(landmark_points[start_idx])
                end = tuple(landmark_points[end_idx])
                cv2.line(frame, start, end, (255, 255, 255), 2)
        
        # Draw landmark points
        for point in landmark_points:
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    
    def draw_debug_overlay(self, frame, debug_image, prediction_probs):
        """Draw debug visualization showing what the model sees."""
        if debug_image is None:
            return
        
        # Scale up debug image
        debug_display = cv2.resize(debug_image, (140, 140),
                                   interpolation=cv2.INTER_NEAREST)
        debug_display = cv2.cvtColor(debug_display, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_display, (0, 0), (139, 139), (0, 255, 0), 2)
        
        frame[10:150, 10:150] = debug_display
        cv2.putText(frame, "Model Input", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show top 3 predictions
        if prediction_probs is not None:
            top_indices = np.argsort(prediction_probs)[-3:][::-1]
            y_start = 200
            
            for i, idx in enumerate(top_indices):
                letter = ASL_LETTERS[idx]
                prob = prediction_probs[idx]
                
                bar_width = int(prob * 100)
                cv2.rectangle(frame, (10, y_start + i*25),
                            (10 + bar_width, y_start + i*25 + 18),
                            (0, 255, 0), -1)
                cv2.putText(frame, f"{letter}: {prob:.0%}",
                           (15, y_start + i*25 + 14),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def process_frame(self, frame):
        """
        Process a single frame using the new MediaPipe Tasks API.
        
        The key difference here is how we create and pass the image:
        - Old API: Just pass the RGB numpy array directly
        - New API: Wrap it in mp.Image with proper format specification
        """
        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width = frame.shape[:2]
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object (required by new Tasks API)
        # This wraps the numpy array with format metadata
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands using the new API
        # Returns a HandLandmarkerResult object
        detection_result = self.hand_landmarker.detect(mp_image)
        
        prediction_probs = None
        
        # Check if any hands were detected
        # New API: results are in detection_result.hand_landmarks (list of hands)
        if detection_result.hand_landmarks:
            # Process first detected hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Preprocess for model
            model_input, debug_image, landmark_points = self.preprocessor.process(
                frame, hand_landmarks, frame_width, frame_height
            )
            
            self.last_debug_image = debug_image
            
            if model_input is not None:
                # Get prediction
                letter, confidence, prediction_probs = self.predict_letter(model_input)
                smoothed_letter = self.smooth_prediction(letter, confidence)
                
                # Calculate bounding box from landmarks for display
                x_coords = [int(lm.x * frame_width) for lm in hand_landmarks]
                y_coords = [int(lm.y * frame_height) for lm in hand_landmarks]
                x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
                y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
                
                # Draw prediction text
                label = f"{smoothed_letter} ({confidence:.0%})"
                cv2.putText(frame, label, (x_min, y_min - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                             (0, 255, 0), 2)
            
            # Draw hand landmarks
            if landmark_points is not None:
                self.draw_landmarks_manual(frame, landmark_points)
        
        # Draw debug overlay
        if self.show_debug and self.last_debug_image is not None:
            self.draw_debug_overlay(frame, self.last_debug_image, prediction_probs)
        
        return frame
    
    def cleanup(self):
        """Release MediaPipe resources."""
        self.hand_landmarker.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Ensure we have the hand landmarker model
    if not ensure_hand_landmarker_model():
        return
    
    # Check for ASL model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: ASL model not found at '{MODEL_PATH}'")
        print("Please train the model first or update MODEL_PATH.")
        return
    
    # Initialize recognizer
    recognizer = ASLRecognizer(MODEL_PATH)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)#, cv2.CAP_V4L2) # Linux ONLY
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Linux ONLY
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n" + "=" * 50)
    print("ASL Recognition (Modern MediaPipe API)")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'd' - Toggle debug view")
    print("=" * 50 + "\n")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Process frame
            annotated = recognizer.process_frame(frame)
            
            # Display
            cv2.imshow('ASL Recognition', annotated)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, annotated)
                print(f"Saved: {filename}")
            elif key == ord('d'):
                recognizer.show_debug = not recognizer.show_debug
                print(f"Debug view: {'ON' if recognizer.show_debug else 'OFF'}")
    
    finally:
        recognizer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()