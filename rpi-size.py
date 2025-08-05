import torch
import torchvision
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import time
import threading
from typing import List, Dict, Tuple

try:
    from picamera2 import Picamera2
except ImportError:
    from fake_picamera2 import Picamera2

class MangoDetector:
    """Mango detection model wrapper"""
    
    def __init__(self, model_path, num_classes=7, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        # Class names mapping
        self.class_names = {
            0: 'background',
            1: 'bruised',
            2: 'not_bruised', 
            3: 'yellow',
            4: 'green_yellow',
            5: 'green',
            6: 'mango'
        }
        
        # Colors for each class (BGR format for OpenCV)
        self.class_colors = {
            1: (0, 0, 255),      # bruised - red
            2: (0, 255, 0),      # not_bruised - green
            3: (0, 255, 255),    # yellow - yellow
            4: (0, 255, 128),    # green_yellow - lime
            5: (0, 128, 0),      # green - dark green
            6: (255, 0, 255)     # mango - magenta
        }
        
        # Load model
        print("Loading mango detection model...")
        self.model = self.load_model(model_path)
        print(f"Model loaded successfully! Using device: {self.device}")
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            
            # Replace classifier head
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, self.num_classes
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict_frame(self, frame):
        """Make predictions on a single frame"""
        if self.model is None:
            return []
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            # Convert RGB to BGR for OpenCV (if needed)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to tensor and normalize
            frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            input_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Extract predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter by confidence
            keep = scores >= self.confidence_threshold
            
            return list(zip(boxes[keep], scores[keep], labels[keep]))
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def draw_predictions(self, frame, predictions):
        """Draw bounding boxes and labels on frame"""
        # Convert PIL Image to numpy array if needed
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for box, score, label in predictions:
            # Extract coordinates
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class info
            class_name = self.class_names.get(label, f'Class_{label}')
            color = self.class_colors.get(label, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f'{class_name}: {score:.2f}'
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return frame

class CameraManager:
    """Enhanced camera manager with mango detection"""
    
    def __init__(self, resolution={'length': 1920, 'width': 1080}, model_path='mango_detection_model.pth'):
        self.resolution = resolution
        self.picam2 = Picamera2()
        self.detection_enabled = False
        self.save_detections = False
        self.detection_count = 0
        
        # Initialize mango detector
        try:
            self.detector = MangoDetector(model_path)
            print("Mango detector initialized successfully")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.detector = None
        
        # Camera initialization
        try:
            self.camera_config = self.picam2.create_video_configuration(
                main={"size": (self.resolution['length'], self.resolution['width'])}
            )
            self.picam2.configure(self.camera_config)
            self.picam2.start()
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.picam2 = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
    
    def get_image(self):
        """Get PIL Image from camera"""
        if self.picam2 is None:
            return None
        
        image = self.picam2.capture_array()
        image = Image.fromarray(image).convert("RGB")
        return image
    
    def capture_array(self):
        """Get numpy array from camera"""
        if self.picam2 is None:
            return None
        
        arr = self.picam2.capture_array()
        return arr
    
    def set_controller_vars(self, app, video_canvas):
        """Set GUI controller variables"""
        self.app = app
        self.video_canvas = video_canvas
    
    def toggle_detection(self):
        """Toggle mango detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ON" if self.detection_enabled else "OFF"
        print(f"Mango detection: {status}")
        return self.detection_enabled
    
    def set_confidence_threshold(self, threshold):
        """Set detection confidence threshold"""
        if self.detector:
            self.detector.confidence_threshold = threshold
            print(f"Confidence threshold set to: {threshold:.2f}")
    
    def toggle_save_detections(self):
        """Toggle saving detection images"""
        self.save_detections = not self.save_detections
        status = "ON" if self.save_detections else "OFF"
        print(f"Save detections: {status}")
        return self.save_detections
    
    def get_video_feed(self):
        """Enhanced video feed with optional mango detection"""
        vid_params = {'f_length': 300, 'f_width': 200, 'buffer': 10, 'x': 0, 'y': 0}
        
        # Get frame from camera
        frame = self.get_image()
        if frame is None:
            self.app.after(vid_params['buffer'], self.get_video_feed)
            return
        
        # Convert to numpy array for processing
        frame_array = np.array(frame)
        
        # Perform mango detection if enabled
        detections = []
        if self.detection_enabled and self.detector:
            detections = self.detector.predict_frame(frame_array)
            
            if detections:
                # Draw detection boxes on frame
                frame_with_boxes = self.detector.draw_predictions(frame_array.copy(), detections)
                
                # Convert back to PIL Image
                frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_with_boxes)
                
                # Save detection if enabled
                if self.save_detections:
                    self.save_detection_image(frame, detections)
        
        # Calculate and draw FPS
        self.update_fps()
        if self.detection_enabled:
            frame = self.add_info_overlay(frame, len(detections))
        
        # Resize for display
        frame = frame.resize((vid_params['f_length'], vid_params['f_width']))
        frame = ImageTk.PhotoImage(frame)
        
        # Update canvas
        self.video_canvas.create_image(vid_params['x'], vid_params['y'], anchor=ctk.NW, image=frame)
        self.video_canvas.image = frame
        
        # Schedule next frame
        self.app.after(vid_params['buffer'], self.get_video_feed)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def add_info_overlay(self, pil_image, detection_count):
        """Add FPS and detection info overlay to PIL image"""
        # Convert PIL to OpenCV
        cv_image = np.array(pil_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Add FPS text
        cv2.putText(cv_image, f'FPS: {self.current_fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection count
        cv2.putText(cv_image, f'Mangoes: {detection_count}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection status
        status = "DETECTING" if self.detection_enabled else "DETECTION OFF"
        color = (0, 255, 0) if self.detection_enabled else (0, 0, 255)
        cv2.putText(cv_image, status, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Convert back to PIL
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image)
    
    def save_detection_image(self, image, detections):
        """Save image with detections"""
        timestamp = int(time.time())
        filename = f"mango_detection_{timestamp}_{len(detections)}mangoes.jpg"
        
        # Convert PIL to OpenCV format for saving
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filename, cv_image)
        self.detection_count += 1
        print(f"Saved detection image: {filename}")
    
    def capture_high_res_detection(self):
        """Capture high resolution image with detections"""
        if self.picam2 is None or self.detector is None:
            print("Camera or detector not available")
            return None
        
        # Capture high resolution image
        full_res_array = self.capture_array()
        
        # Perform detection
        detections = self.detector.predict_frame(full_res_array)
        
        if detections:
            # Draw detections on high-res image
            annotated_image = self.detector.draw_predictions(full_res_array.copy(), detections)
            
            # Save high-res detection
            timestamp = int(time.time())
            filename = f"highres_mango_detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_image)
            
            print(f"High-res detection saved: {filename}")
            print(f"Found {len(detections)} mangoes")
            
            # Print detection details
            for i, (box, score, label) in enumerate(detections):
                class_name = self.detector.class_names.get(label, f'Class_{label}')
                print(f"  Mango {i+1}: {class_name} ({score:.2f} confidence)")
            
            return filename
        else:
            print("No mangoes detected in high-res capture")
            return None
    
    def stop_camera(self):
        """Stop the camera"""
        if self.picam2:
            self.picam2.stop()
            print("Camera stopped")

class MangoDetectionApp(ctk.CTk):
    """CustomTkinter GUI application for mango detection"""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Raspberry Pi Mango Detection System")
        self.geometry("800x600")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        
        # Create GUI
        self.create_widgets()
        
        # Start video feed
        if self.camera_manager.picam2:
            self.camera_manager.set_controller_vars(self, self.video_canvas)
            self.camera_manager.get_video_feed()
    
    def create_widgets(self):
        """Create GUI widgets"""
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video canvas
        self.video_canvas = ctk.CTkCanvas(self.main_frame, width=300, height=200, bg="black")
        self.video_canvas.pack(pady=10)
        
        # Control buttons frame
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Detection toggle button
        self.detection_button = ctk.CTkButton(
            self.controls_frame, 
            text="Start Detection", 
            command=self.toggle_detection
        )
        self.detection_button.pack(side="left", padx=5)
        
        # High-res capture button
        self.capture_button = ctk.CTkButton(
            self.controls_frame, 
            text="High-Res Capture", 
            command=self.capture_high_res
        )
        self.capture_button.pack(side="left", padx=5)
        
        # Save toggle button
        self.save_button = ctk.CTkButton(
            self.controls_frame, 
            text="Auto-Save OFF", 
            command=self.toggle_save
        )
        self.save_button.pack(side="left", padx=5)
        
        # Settings frame
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Confidence threshold slider
        self.confidence_label = ctk.CTkLabel(self.settings_frame, text="Confidence Threshold:")
        self.confidence_label.pack(side="left", padx=5)
        
        self.confidence_slider = ctk.CTkSlider(
            self.settings_frame, 
            from_=0.1, 
            to=1.0, 
            number_of_steps=18,
            command=self.update_confidence
        )
        self.confidence_slider.set(0.5)
        self.confidence_slider.pack(side="left", padx=5)
        
        self.confidence_value = ctk.CTkLabel(self.settings_frame, text="0.5")
        self.confidence_value.pack(side="left", padx=5)
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame, 
            text="Status: Camera ready. Click 'Start Detection' to begin."
        )
        self.status_label.pack(pady=5)
    
    def toggle_detection(self):
        """Toggle mango detection"""
        if self.camera_manager:
            is_enabled = self.camera_manager.toggle_detection()
            self.detection_button.configure(
                text="Stop Detection" if is_enabled else "Start Detection"
            )
            status = "Detection ON" if is_enabled else "Detection OFF"
            self.status_label.configure(text=f"Status: {status}")
    
    def toggle_save(self):
        """Toggle auto-save detections"""
        if self.camera_manager:
            is_enabled = self.camera_manager.toggle_save_detections()
            self.save_button.configure(
                text="Auto-Save ON" if is_enabled else "Auto-Save OFF"
            )
    
    def capture_high_res(self):
        """Capture high resolution image with detections"""
        if self.camera_manager:
            self.status_label.configure(text="Status: Capturing high-res image...")
            self.update()  # Force GUI update
            
            # Run capture in thread to avoid GUI freeze
            def capture_thread():
                filename = self.camera_manager.capture_high_res_detection()
                if filename:
                    self.after(0, lambda: self.status_label.configure(
                        text=f"Status: High-res image saved as {filename}"))
                else:
                    self.after(0, lambda: self.status_label.configure(
                        text="Status: No mangoes detected in high-res capture"))
            
            threading.Thread(target=capture_thread, daemon=True).start()
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        if self.camera_manager:
            self.camera_manager.set_confidence_threshold(float(value))
            self.confidence_value.configure(text=f"{float(value):.1f}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.camera_manager:
            self.camera_manager.stop_camera()
        self.destroy()

def main():
    """Main function to run the application"""
    
    # Check if model exists
    import os
    model_path = 'mango_detection_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Make sure you have trained the model first.")
        return
    
    # Create and run app
    app = MangoDetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
