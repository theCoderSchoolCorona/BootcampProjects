"""
    
Controls:
    - Press letter key (a-y, except j) to capture sample for that letter
    - Press SPACE to capture sample for current selected letter
    - Press UP/DOWN arrows to change selected letter
    - Press 'q' to quit and save dataset
    - Press 'r' to review collected samples

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

OUTPUT_DIR = "collected_samples"

# MediaPipe model paths and URLs
HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'
HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Detection thresholds
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7

# ASL letters (J and Z excluded - they require motion)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

LETTER_TO_LABEL = {letter: i for i, letter in enumerate(ASL_LETTERS)}


# =============================================================================
# MODEL DOWNLOAD HELPER
# =============================================================================

def ensure_hand_landmarker_model():
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print(f"Downloading hand landmarker model...")
        print(f"  URL: {HAND_LANDMARKER_MODEL_URL}")
        
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
            print("  Download complete!")
        except Exception as e:
            print(f"  Error: {e}")
            return False
    
    return True


# =============================================================================
# HAND PREPROCESSOR
# =============================================================================

class HandPreprocessor:
    """
    Preprocesses hand images to match Sign MNIST format.
    
    This class handles the domain gap between your camera's raw output
    and the controlled conditions of the Sign MNIST dataset by:
    1. Segmenting the hand from background using landmarks + skin detection
    2. Centering the hand in the frame
    3. Applying consistent contrast enhancement
    """
    
    def __init__(self, target_size=28):
        self.target_size = target_size
        
        # YCrCb skin detection thresholds (works across various skin tones)
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)
    
    def create_hand_mask(self, frame, hand_landmarks, frame_width, frame_height):
        """
        Create a binary mask to isolate the hand from the background.
        
        Combines two techniques:
        1. Convex hull around MediaPipe landmarks (captures hand shape)
        2. Skin color detection (refines edges)
        """
        h, w = frame_height, frame_width
        
        # Convert normalized landmarks to pixel coordinates
        # The new API returns NormalizedLandmark objects with x, y in [0, 1] range
        landmark_points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            landmark_points.append([px, py])
        
        landmark_points = np.array(landmark_points, dtype=np.int32)
        
        # Create convex hull mask
        hull = cv2.convexHull(landmark_points)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        
        # Dilate to ensure we capture fingertips fully
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        hull_mask = cv2.dilate(hull_mask, kernel, iterations=1)
        
        # Skin color detection in YCrCb space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        
        # Clean up skin mask
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_small)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Combine: hull defines region, skin refines edges
        combined_mask = cv2.bitwise_and(hull_mask, skin_mask)
        
        # Fall back to hull only if skin detection fails (unusual lighting)
        if cv2.countNonZero(combined_mask) < 1000:
            combined_mask = hull_mask
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask, landmark_points
    
    def get_centered_square_crop(self, mask, frame_gray, padding_ratio=0.15):
        """
        Extract a square crop centered on the hand's center of mass.
        
        This ensures consistent positioning like Sign MNIST samples,
        where hands are well-centered in the frame.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find center of mass
        moments = cv2.moments(largest_contour)
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Get bounding rect for size reference
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Make it square using larger dimension plus padding
        size = int(max(w, h) * (1 + padding_ratio * 2))
        half_size = size // 2
        
        frame_h, frame_w = frame_gray.shape
        
        # Calculate crop boundaries centered on center of mass
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
            processed: The 28x28 preprocessed image ready for training
            raw_masked: Intermediate masked image for debugging
            landmark_points: Pixel coordinates of landmarks for drawing
        """
        # Create segmentation mask
        mask, landmark_points = self.create_hand_mask(
            frame, hand_landmarks, frame_width, frame_height
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        crop_result = self.get_centered_square_crop(mask, gray)
        if crop_result is None:
            return None, None, landmark_points
        
        hand_crop, mask_crop, bbox = crop_result
        
        if hand_crop.size == 0:
            return None, None, landmark_points
        
        # Apply mask (white background to match Sign MNIST style)
        masked_hand = hand_crop.copy()
        masked_hand[mask_crop == 0] = 255
        
        # CLAHE for consistent contrast across different lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked_hand)
        
        # Resize to model input size
        resized = cv2.resize(enhanced, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_AREA)
        
        # Light blur to match Sign MNIST smoothness
        processed = cv2.GaussianBlur(resized, (3, 3), 0)
        
        return processed, masked_hand, landmark_points


# =============================================================================
# DATA COLLECTOR (using new MediaPipe Tasks API)
# =============================================================================

class ASLDataCollector:
    """
    Collects and manages ASL training samples.
    
    Uses the modern MediaPipe Tasks API for hand detection, which requires:
    1. Creating a HandLandmarker with explicit options
    2. Wrapping frames in mp.Image objects
    3. Handling HandLandmarkerResult objects for detection output
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Hand Landmarker (new Tasks API)
        print("Initializing MediaPipe Hand Landmarker...")
        
        base_options = mp_tasks.BaseOptions(
            model_asset_path=HAND_LANDMARKER_MODEL_PATH
        )
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("Hand Landmarker initialized!")
        
        # Preprocessor
        self.preprocessor = HandPreprocessor()
        
        # Collection state
        self.samples = {letter: [] for letter in ASL_LETTERS}
        self.selected_letter_idx = 0
        
        # Load any existing samples
        self.load_existing_samples()
    
    def load_existing_samples(self):
        """Load previously collected samples from disk."""
        for letter in ASL_LETTERS:
            letter_dir = self.output_dir / letter
            if letter_dir.exists():
                samples = list(letter_dir.glob("*.png"))
                self.samples[letter] = samples
                if samples:
                    print(f"Loaded {len(samples)} existing samples for '{letter}'")
    
    def save_sample(self, letter, processed_image):
        """Save a collected sample to disk."""
        letter_dir = self.output_dir / letter
        letter_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{letter}_{timestamp}.png"
        filepath = letter_dir / filename
        
        cv2.imwrite(str(filepath), processed_image)
        self.samples[letter].append(filepath)
        
        return filepath
    
    def export_to_csv(self, output_path="custom_training_data.csv"):
        """
        Export all samples to CSV format matching Sign MNIST.
        
        Format: label, pixel0, pixel1, ..., pixel783
        Each row is one sample with 784 pixel values (28x28 flattened).
        """
        import pandas as pd
        
        rows = []
        for letter, sample_paths in self.samples.items():
            label = LETTER_TO_LABEL[letter]
            
            for path in sample_paths:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                pixels = img.flatten().tolist()
                row = [label] + pixels
                rows.append(row)
        
        columns = ['label'] + [f'pixel{i}' for i in range(784)]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_path, index=False)
        
        print(f"Exported {len(rows)} samples to {output_path}")
        return output_path
    
    @property
    def selected_letter(self):
        return ASL_LETTERS[self.selected_letter_idx]
    
    def next_letter(self):
        self.selected_letter_idx = (self.selected_letter_idx + 1) % len(ASL_LETTERS)
    
    def prev_letter(self):
        self.selected_letter_idx = (self.selected_letter_idx - 1) % len(ASL_LETTERS)
    
    def select_letter(self, letter):
        letter = letter.upper()
        if letter in ASL_LETTERS:
            self.selected_letter_idx = ASL_LETTERS.index(letter)
            return True
        return False
    
    def draw_landmarks_manual(self, frame, landmark_points):
        """Draw hand skeleton since new API needs manual drawing."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start = tuple(landmark_points[start_idx])
                end = tuple(landmark_points[end_idx])
                cv2.line(frame, start, end, (255, 255, 255), 2)
        
        for point in landmark_points:
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    
    def process_frame(self, frame):
        """
        Process a frame using the new MediaPipe Tasks API.
        
        Key differences from legacy API:
        1. Frame must be wrapped in mp.Image object
        2. detect() returns HandLandmarkerResult
        3. Results are in detection_result.hand_landmarks list
        """
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB and wrap in MediaPipe Image (required by Tasks API)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        detection_result = self.hand_landmarker.detect(mp_image)
        
        processed_image = None
        landmark_points = None
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Preprocess
            processed, raw_masked, landmark_points = self.preprocessor.process(
                frame, hand_landmarks, frame_width, frame_height
            )
            
            if processed is not None:
                processed_image = processed
                
                # Show preview in corner
                preview = cv2.resize(processed, (140, 140),
                                    interpolation=cv2.INTER_NEAREST)
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(preview, (0, 0), (139, 139), (0, 255, 0), 2)
                frame[10:150, 10:150] = preview
            
            # Draw landmarks
            if landmark_points is not None:
                self.draw_landmarks_manual(frame, landmark_points)
        
        # Draw UI overlay
        self.draw_ui(frame)
        
        return frame, processed_image
    
    def draw_ui(self, frame):
        """Draw the data collection UI overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        panel_x = w - 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Data Collection", (panel_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Selected letter
        cv2.putText(frame, f"Letter: {self.selected_letter}", (panel_x + 10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Sample count
        count = len(self.samples[self.selected_letter])
        cv2.putText(frame, f"Samples: {count}", (panel_x + 10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        y = 150
        instructions = [
            "Controls:",
            "SPACE - Capture",
            "A-Y   - Capture for letter",
            "UP/DN - Change letter",
            "R     - Review samples",
            "Q     - Quit & save",
        ]
        
        for inst in instructions:
            cv2.putText(frame, inst, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 20
        
        # Sample counts grid
        y = 300
        cv2.putText(frame, "Collected:", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        
        for i, letter in enumerate(ASL_LETTERS):
            count = len(self.samples[letter])
            color = (0, 255, 0) if count > 0 else (100, 100, 100)
            
            col = i % 6
            row = i // 6
            
            x = panel_x + 10 + col * 30
            y_pos = y + row * 25
            
            text = f"{letter}:{count}"
            cv2.putText(frame, text, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def cleanup(self):
        """Release MediaPipe resources."""
        self.hand_landmarker.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Ensure model is downloaded
    if not ensure_hand_landmarker_model():
        return
    
    collector = ASLDataCollector(OUTPUT_DIR)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n" + "=" * 50)
    print("ASL Data Collection (Modern MediaPipe API)")
    print("=" * 50)
    print(f"Saving samples to: {OUTPUT_DIR}/")
    print("Show your hand sign and press SPACE or letter key to capture.")
    print("Aim for 30-50 samples per letter for best results.")
    print("=" * 50 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, processed = collector.process_frame(frame)
            cv2.imshow('ASL Data Collection', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                if processed is not None:
                    path = collector.save_sample(collector.selected_letter, processed)
                    print(f"Saved sample for '{collector.selected_letter}': {path}")
                else:
                    print("No hand detected - cannot capture")
            
            elif key == 82 or key == 0:  # Up arrow
                collector.prev_letter()
            
            elif key == 84 or key == 1:  # Down arrow
                collector.next_letter()
            
            elif key == ord('r'):
                print("\nSample counts:")
                for letter, samples in collector.samples.items():
                    print(f"  {letter}: {len(samples)}")
                print()
            
            # Letter keys (a-y, except j)
            elif 97 <= key <= 121:
                letter = chr(key).upper()
                if letter in ASL_LETTERS and processed is not None:
                    path = collector.save_sample(letter, processed)
                    print(f"Saved sample for '{letter}': {path}")
    
    finally:
        print("\n" + "=" * 50)
        print("Collection Summary:")
        total = 0
        for letter in ASL_LETTERS:
            count = len(collector.samples[letter])
            total += count
            if count > 0:
                print(f"  {letter}: {count} samples")
        print(f"\nTotal: {total} samples")
        
        if total > 0:
            print("\nExporting to CSV...")
            collector.export_to_csv()
        
        collector.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()