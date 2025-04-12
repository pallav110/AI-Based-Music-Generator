# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import mediapipe as mp
from deepface import DeepFace
import time
import queue
import logging
import json
import argparse
import os
from collections import deque
import signal
import datetime
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emotion_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EmotionDetector")


class Config:
    """Configuration class with default settings that can be overridden via command line arguments"""
    def __init__(self):
        # Detection settings
        self.FACE_DETECTION_INTERVAL = 0.2
        self.DISPLAY_MESH = True
        self.SINGLE_FACE_MODE = True
        self.ACTIONS = ['emotion']
        self.USE_THREADS = True
        self.SMOOTH_EMOTIONS = True
        self.USE_BACKUP_DETECTOR = True
        self.MAX_THREAD_QUEUE = 2
        self.MAX_FPS_HISTORY = 30
        self.EMOTION_SMOOTHING_FACTOR = 0.7
        self.EMOTION_HISTORY_SIZE = 100
        self.DETECTION_CONFIDENCE = 0.5
        self.TRACKING_CONFIDENCE = 0.5

        # Display settings
        self.TEXT_COLOR = (255, 255, 255)
        self.MENU_COLOR = (0, 255, 255)  # Yellow for menu
        self.RECT_COLOR = (0, 255, 0)
        self.BACKGROUND_ALPHA = 0.4  # Transparency for text background
        self.DISPLAY_SCALE = 1.0  # Scale factor for display

        # Camera settings
        self.CAMERA_ID = 0
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.CAMERA_FPS = 30

        # Model settings
        self.MODEL_BACKEND = 'opencv'
        
        # Statistics settings
        self.STATS_DIR = "stats"
        self.STATS_FILE = os.path.join(self.STATS_DIR, "emotion_stats.json")
        self.CSV_FILE = os.path.join(self.STATS_DIR, "emotion_timeline.csv")
        self.AUTO_SAVE_INTERVAL = 60  # Auto-save stats every 60 seconds

    def parse_args(self):
        """Parse command line arguments and update the configuration"""
        parser = argparse.ArgumentParser(description="Emotion Detection System")
        parser.add_argument("--camera", type=int, default=self.CAMERA_ID, help="Camera device ID")
        parser.add_argument("--width", type=int, default=self.CAMERA_WIDTH, help="Camera width")
        parser.add_argument("--height", type=int, default=self.CAMERA_HEIGHT, help="Camera height")
        parser.add_argument("--fps", type=int, default=self.CAMERA_FPS, help="Camera FPS")
        parser.add_argument("--scale", type=float, default=self.DISPLAY_SCALE, help="Display scale factor")
        parser.add_argument("--backend", type=str, default=self.MODEL_BACKEND, 
                            choices=["opencv", "ssd", "mtcnn", "retinaface"], 
                            help="Face detection backend")
        parser.add_argument("--no-mesh", action="store_false", dest="display_mesh", 
                            help="Disable face mesh display")
        parser.add_argument("--multi-face", action="store_false", dest="single_face", 
                            help="Enable multi-face detection")
        parser.add_argument("--detection-confidence", type=float, default=self.DETECTION_CONFIDENCE,
                            help="MediaPipe face detection confidence threshold")
        parser.add_argument("--smoothing", type=float, default=self.EMOTION_SMOOTHING_FACTOR,
                            help="Emotion smoothing factor (0-1)")
        parser.add_argument("--auto-save", type=int, default=self.AUTO_SAVE_INTERVAL,
                            help="Auto-save interval in seconds (0 to disable)")
        
        args = parser.parse_args()
        
        # Update config with parsed arguments
        self.CAMERA_ID = args.camera
        self.CAMERA_WIDTH = args.width
        self.CAMERA_HEIGHT = args.height
        self.CAMERA_FPS = args.fps
        self.MODEL_BACKEND = args.backend
        self.DISPLAY_MESH = args.display_mesh
        self.SINGLE_FACE_MODE = args.single_face
        self.DISPLAY_SCALE = args.scale
        self.DETECTION_CONFIDENCE = args.detection_confidence
        self.EMOTION_SMOOTHING_FACTOR = args.smoothing
        self.AUTO_SAVE_INTERVAL = args.auto_save
        
        return self


class EmotionDetector:
    """Main class for emotion detection using webcam"""
    
    def __init__(self, config=None):
        """Initialize the emotion detector with the given or default configuration"""
        self.config = config if config else Config()
        
        # Initialize state variables
        self.cap = None
        self.latest_results = None
        self.processing_frame = False
        self.detection_active = True
        self.emotion_history = deque(maxlen=self.config.EMOTION_HISTORY_SIZE)
        self.emotion_smoothed = {}
        self.emotion_timeline = []
        self.fps_history = deque(maxlen=self.config.MAX_FPS_HISTORY)
        self.thread_queue = queue.Queue(maxsize=self.config.MAX_THREAD_QUEUE)
        self.paused = False
        self.show_debug = False
        self.display_mesh = self.config.DISPLAY_MESH
        self.current_emotion = None
        self.last_save_time = time.time()
        self.running = True
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)  # Track processing times
        
        # Ensure stats directory exists
        os.makedirs(self.config.STATS_DIR, exist_ok=True)
        
        # Setup components
        self._setup_camera()
        self._setup_mediapipe()
        self._setup_signal_handlers()
        
        # Load existing stats if available
        self._load_stats()

    def _setup_camera(self):
        """Initialize the camera with configured settings"""
        logger.info(f"Opening webcam (ID: {self.config.CAMERA_ID})...")
        self.cap = cv2.VideoCapture(self.config.CAMERA_ID, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera. Check connection and ID.")
            # Try fallback to default camera if specified camera fails
            if self.config.CAMERA_ID != 0:
                logger.info("Trying fallback to default camera (ID: 0)...")
                self.config.CAMERA_ID = 0
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("Camera initialization failed")
            else:
                raise RuntimeError("Camera initialization failed")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Get actual camera properties (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

    def _setup_mediapipe(self):
        """Setup MediaPipe for facial landmark detection"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Initialize face mesh with configured settings
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1 if self.config.SINGLE_FACE_MODE else 2,
                min_detection_confidence=self.config.DETECTION_CONFIDENCE,
                min_tracking_confidence=self.config.TRACKING_CONFIDENCE
            )
            self.mediapipe_available = True
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            self.mediapipe_available = False
            logger.warning(f"MediaPipe not available: {e}")
            logger.warning("Face mesh visualization will be disabled")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        self.running = False

    def _load_stats(self):
        """Load existing statistics if available"""
        try:
            if os.path.exists(self.config.STATS_FILE):
                with open(self.config.STATS_FILE, 'r') as f:
                    stats = json.load(f)
                    logger.info(f"Loaded existing statistics from {self.config.STATS_FILE}")
                    
                    # Initialize with previous emotions if available
                    if "counts" in stats:
                        for emotion, count in stats["counts"].items():
                            # Add previous emotions to history with appropriate weighting
                            for _ in range(min(count, self.config.EMOTION_HISTORY_SIZE // 10)):
                                self.emotion_history.append(emotion)
        except Exception as e:
            logger.warning(f"Failed to load existing statistics: {e}")

    def analyze_face(self, frame):
        """Analyze face for emotion in a separate thread"""
        start_time = time.time()
        try:
            # Run DeepFace emotion analysis
            results = DeepFace.analyze(
                img_path=frame, 
                actions=self.config.ACTIONS, 
                enforce_detection=False,
                detector_backend=self.config.MODEL_BACKEND, 
                silent=True
            )
            
            if results:
                # Store results and update emotion history
                self.latest_results = results
                dominant_emotion = results[0]['dominant_emotion']
                self.emotion_history.append(dominant_emotion)
                
                # Update smoothed emotion probabilities
                if self.config.SMOOTH_EMOTIONS:
                    self._smooth_emotions(results[0]['emotion'])
                    
                # Set current emotion based on smoothed values if available
                if self.emotion_smoothed:
                    self.current_emotion = max(self.emotion_smoothed, key=self.emotion_smoothed.get)
                else:
                    self.current_emotion = dominant_emotion
                
                # Add to timeline for tracking changes over time
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.emotion_timeline.append({
                    "timestamp": timestamp,
                    "emotion": self.current_emotion,
                    "values": self.emotion_smoothed.copy() if self.emotion_smoothed else results[0]['emotion']
                })
                
                if self.show_debug:
                    logger.debug(f"Detected emotion: {dominant_emotion}")
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            self.latest_results = None
        finally:
            self.processing_frame = False  # Allow new frames to be processed
            
            # Track processing time for performance monitoring
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if self.show_debug and len(self.processing_times) > 5:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logger.debug(f"Avg emotion processing time: {avg_time:.3f}s")

    def _smooth_emotions(self, emotions):
        """Apply exponential smoothing to emotion probabilities"""
        if not self.emotion_smoothed:
            # Initialize with first reading
            self.emotion_smoothed = emotions.copy()
        else:
            # Apply exponential smoothing
            alpha = self.config.EMOTION_SMOOTHING_FACTOR
            for emotion, value in emotions.items():
                self.emotion_smoothed[emotion] = alpha * value + (1 - alpha) * self.emotion_smoothed.get(emotion, 0)

    def get_emotion_stats(self):
        """Calculate emotion statistics from history"""
        if not self.emotion_history:
            return {}
            
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        total = len(self.emotion_history)
        emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        
        # Calculate additional statistics
        session_duration = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        
        return {
            "counts": emotion_counts,
            "percentages": emotion_percentages,
            "dominant": max(emotion_counts, key=emotion_counts.get) if emotion_counts else None,
            "total_samples": total,
            "session_duration": session_duration,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_fps": self.get_average_fps(),
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        }

    def export_timeline_to_csv(self):
        """Export the emotion timeline to a CSV file"""
        if not self.emotion_timeline:
            logger.warning("No emotion timeline data to export")
            return False
            
        try:
            with open(self.config.CSV_FILE, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'emotion'] + list(self.emotion_timeline[0]['values'].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for entry in self.emotion_timeline:
                    row = {
                        'timestamp': entry['timestamp'],
                        'emotion': entry['emotion']
                    }
                    for emotion, value in entry['values'].items():
                        row[emotion] = value
                    writer.writerow(row)
                    
            logger.info(f"Emotion timeline exported to {self.config.CSV_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to export timeline: {e}")
            return False

    def save_statistics(self):
        """Save detected emotions to a JSON file"""
        stats = self.get_emotion_stats()
        if not stats:
            logger.warning("No emotions detected yet, nothing to save.")
            return False
            
        with open(self.config.STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)

        logger.info(f"Emotion statistics saved to {self.config.STATS_FILE}")
        self.last_save_time = time.time()
        
        # Also export timeline to CSV
        self.export_timeline_to_csv()
        
        return True

    def calculate_fps(self, prev_time):
        """Calculate the FPS and maintain history"""
        current_time = time.time()
        time_diff = current_time - prev_time
        
        # Avoid division by zero
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_history.append(fps)
        else:
            fps = 0
            
        return fps, current_time

    def get_average_fps(self):
        """Get the average FPS from history"""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    def draw_menu(self, frame, fps):
        """Display control instructions & FPS on screen with semi-transparent background"""
        menu_text = [
            "Press 'q' or ESC to Quit",
            "Press 'p' to Pause/Resume",
            "Press 'd' to Toggle Debug Info",
            "Press 'm' to Toggle Face Mesh",
            "Press 's' to Save Statistics",
            "Press 'c' to Export CSV",
            f"FPS: {fps:.1f}",
            f"Current: {self.current_emotion}" if self.current_emotion else "No emotion detected"
        ]

        # Create semi-transparent overlay for menu
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 30 + len(menu_text) * 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.BACKGROUND_ALPHA, frame, 1 - self.config.BACKGROUND_ALPHA, 0, frame)

        # Draw menu text
        y_offset = 25
        for line in menu_text:
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, self.config.MENU_COLOR, 1, cv2.LINE_AA)
            y_offset += 20

    def draw_emotion_bar(self, frame):
        """Draw a bar graph of emotion probabilities"""
        if not self.latest_results or not self.emotion_smoothed:
            return
            
        emotions = self.emotion_smoothed
        
        # Setup the drawing area
        bar_height = 24
        gap = 2
        max_bar_width = 150
        total_height = (bar_height + gap) * len(emotions)
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (frame.shape[1] - max_bar_width - 110, frame.shape[0] - total_height - 10),
                     (frame.shape[1] - 10, frame.shape[0] - 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.BACKGROUND_ALPHA, frame, 1 - self.config.BACKGROUND_ALPHA, 0, frame)
        
        # Draw each emotion bar
        y = frame.shape[0] - total_height - 5
        
        # Sort emotions by value
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, value in sorted_emotions:
            # Determine bar color based on emotion
            if emotion == 'happy':
                color = (0, 255, 0)     # Green
            elif emotion == 'sad':
                color = (255, 0, 0)     # Blue
            elif emotion == 'angry':
                color = (0, 0, 255)     # Red
            elif emotion == 'fear':
                color = (255, 0, 255)   # Purple
            elif emotion == 'surprise':
                color = (0, 255, 255)   # Yellow
            elif emotion == 'disgust':
                color = (128, 0, 128)   # Purple
            else:
                color = (200, 200, 200) # Gray
                
            # Draw emotion label
            cv2.putText(frame, emotion, (frame.shape[1] - max_bar_width - 100, y + bar_height - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw bar
            bar_width = int(value * max_bar_width / 100)
            cv2.rectangle(frame, 
                        (frame.shape[1] - max_bar_width - 10, y), 
                        (frame.shape[1] - max_bar_width - 10 + bar_width, y + bar_height), 
                        color, -1)
            
            # Draw percentage
            percentage_text = f"{value:.1f}%"
            cv2.putText(frame, percentage_text, 
                      (frame.shape[1] - max_bar_width + bar_width - 5, y + bar_height - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            y += bar_height + gap

    def handle_keypress(self, key):
        """Handle user key presses"""
        if key in [ord('q'), 27]:  # Quit on 'q' or ESC
            return False
        elif key == ord('p'):  # Pause/Resume
            self.paused = not self.paused
            logger.info("Paused" if self.paused else "Resumed")
        elif key == ord('d'):  # Toggle Debug Info
            self.show_debug = not self.show_debug
            logger.info(f"Debug mode {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('m'):  # Toggle Face Mesh
            self.display_mesh = not self.display_mesh
            logger.info(f"Face mesh {'ENABLED' if self.display_mesh else 'DISABLED'}")
        elif key == ord('s'):  # Save statistics
            self.save_statistics()
            logger.info("Statistics saved manually")
        elif key == ord('c'):  # Export to CSV
            self.export_timeline_to_csv()
            logger.info("Timeline exported to CSV manually")
        elif key == ord('f'):  # Toggle fullscreen
            cv2.setWindowProperty("Emotion Detector", cv2.WND_PROP_FULLSCREEN, 
                                 cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("Emotion Detector", cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        
        return True

    def draw_face_box(self, frame):
        """Draw bounding box around detected face"""
        if not self.latest_results:
            return
            
        try:
            face = self.latest_results[0]['region']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.RECT_COLOR, 2)
            
            # Draw emotion label
            if self.current_emotion:
                label = f"{self.current_emotion}"
                # Draw text background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame, 
                    (x, y - text_height - 10), 
                    (x + text_width + 10, y), 
                    self.config.RECT_COLOR, 
                    -1
                )
                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
        except Exception as e:
            if self.show_debug:
                logger.debug(f"Error drawing face box: {e}")

    def check_auto_save(self):
        """Check if it's time to auto-save statistics"""
        if (self.config.AUTO_SAVE_INTERVAL > 0 and 
            time.time() - self.last_save_time > self.config.AUTO_SAVE_INTERVAL):
            self.save_statistics()
            logger.info("Auto-saved statistics")

    def adjust_frame_if_needed(self, frame):
        """Resize frame if display scale is not 1.0"""
        if self.config.DISPLAY_SCALE != 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.config.DISPLAY_SCALE)
            new_width = int(width * self.config.DISPLAY_SCALE)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def process(self):
        """Main loop for emotion detection"""
        prev_time = time.time()
        self.start_time = time.time()
        logger.info("Starting emotion detection. Press 'q' to quit.")

        try:
            while self.running:
                # Check for auto-save
                self.check_auto_save()
                
                # Process pause state
                if self.paused:
                    paused_screen = np.zeros(
                        (self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 3), 
                        dtype=np.uint8
                    )
                    cv2.putText(
                        paused_screen, 
                        "Paused - Press 'p' to Resume", 
                        (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2
                    )
                    cv2.imshow("Emotion Detector", paused_screen)
                    key = cv2.waitKey(100) & 0xFF
                    if not self.handle_keypress(key):
                        break
                    continue

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    if not self.cap.isOpened():
                        logger.error("Camera disconnected")
                        break
                    # Wait a bit and try again
                    time.sleep(0.1)
                    continue

                # Increment frame counter for metrics
                self.frame_count += 1

                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe if available
                if self.mediapipe_available and self.display_mesh:
                    results = self.face_mesh.process(frame_rgb)
                    if results and results.multi_face_landmarks:
                        # Draw face mesh landmarks
                        for face_landmarks in results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, 
                                face_landmarks, 
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )

                # Start emotion analysis in separate thread if not already processing
                # Only process every few frames to reduce CPU load
                if (not self.processing_frame and 
                    self.detection_active and 
                    self.frame_count % max(1, round(self.get_average_fps() * self.config.FACE_DETECTION_INTERVAL)) == 0):
                    self.processing_frame = True
                    threading.Thread(
                        target=self.analyze_face, 
                        args=(frame_rgb.copy(),), 
                        daemon=True
                    ).start()

                # Draw face box and current emotion
                self.draw_face_box(frame)
                
                # Draw emotion bar graph
                self.draw_emotion_bar(frame)

                # Calculate and display FPS
                fps, prev_time = self.calculate_fps(prev_time)
                self.draw_menu(frame, self.get_average_fps())

                # Show debug info if enabled
                if self.show_debug and self.latest_results:
                    debug_y = 80
                    for emotion, value in self.emotion_smoothed.items():
                        cv2.putText(
                            frame, 
                            f"{emotion}: {value:.1f}%", 
                            (10, debug_y), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, 
                            self.config.TEXT_COLOR, 
                            1
                        )
                        debug_y += 20
                    
                    # Add processing performance metrics
                    if self.processing_times:
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        cv2.putText(
                            frame,
                            f"Processing: {avg_time:.3f}s",
                            (10, debug_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            self.config.TEXT_COLOR,
                            1
                        )

                # Resize frame if needed
                display_frame = self.adjust_frame_if_needed(frame)

                # Display frame
                cv2.imshow("Emotion Detector", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keypress(key):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and close windows"""
        logger.info("Shutting down...")
        if self.cap:
            self.cap.release()
        if self.mediapipe_available and hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        cv2.destroyAllWindows()
        
        # Save statistics before exit
        self.save_statistics()
        logger.info(f"Session completed. Processed {self.frame_count} frames.")


class EmotionAnalysisApp:
    """Application wrapper to manage the emotion detector"""
    
    def __init__(self):
        """Initialize the application"""
        self.config = Config().parse_args()
        self.detector = None
        
    def run(self):
        """Run the emotion detection application"""
        try:
            logger.info("Starting Emotion Analysis Application")
            self.detector = EmotionDetector(self.config)
            self.detector.process()
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            return 1
        return 0


if __name__ == "__main__":
    app = EmotionAnalysisApp()
    exit_code = app.run()
    exit(exit_code)