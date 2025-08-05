import cv2
import torch
import torchvision
import numpy as np
import math
from typing import List, Dict, Tuple
import json

class MangoMeasurementSystem:
    def __init__(self, model_path, num_classes=7):
        """
        Initialize the mango measurement system
        
        Args:
            model_path: Path to trained model
            num_classes: Number of classes (6 mango classes + background)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, num_classes)
        
        # Class names
        self.class_names = {
            1: 'bruised', 2: 'not_bruised', 3: 'yellow',
            4: 'green_yellow', 5: 'green', 6: 'mango'
        }
        
        # Calibration settings (you'll need to adjust these)
        self.pixels_per_cm = None  # Will be set during calibration
        self.reference_object_size_cm = None  # Size of reference object in cm
        
    def load_model(self, model_path, num_classes):
        """Load the trained mango detection model"""
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def detect_mangoes(self, image, confidence_threshold=0.5):
        """Detect mangoes in image and return bounding boxes"""
        if self.model is None:
            return []
        
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by confidence
        keep = scores >= confidence_threshold
        
        detections = []
        for i, (box, score, label) in enumerate(zip(boxes[keep], scores[keep], labels[keep])):
            detections.append({
                'id': i,
                'box': box,
                'score': score,
                'label': label,
                'class_name': self.class_names.get(label, f'Class_{label}')
            })
        
        return detections
    
    def calibrate_with_reference_object(self, image, reference_box, reference_size_cm):
        """
        Calibrate the measurement system using a reference object
        
        Args:
            image: Input image
            reference_box: Bounding box of reference object [x1, y1, x2, y2]
            reference_size_cm: Known size of reference object in cm
        """
        x1, y1, x2, y2 = reference_box
        
        # Calculate reference object dimensions in pixels
        ref_width_pixels = x2 - x1
        ref_height_pixels = y2 - y1
        
        # Use the larger dimension for calibration (more accurate)
        ref_size_pixels = max(ref_width_pixels, ref_height_pixels)
        
        # Calculate pixels per cm
        self.pixels_per_cm = ref_size_pixels / reference_size_cm
        self.reference_object_size_cm = reference_size_cm
        
        print(f"Calibration complete:")
        print(f"  Reference object: {reference_size_cm} cm")
        print(f"  Reference pixels: {ref_size_pixels:.1f} pixels")
        print(f"  Scale: {self.pixels_per_cm:.2f} pixels/cm")
        
        return self.pixels_per_cm
    
    def auto_calibrate_with_coin(self, image, coin_type="quarter"):
        """
        Auto-calibrate using a detected coin (if you add coin detection)
        
        Args:
            image: Input image
            coin_type: Type of coin for reference
        """
        # Standard coin sizes (diameter in cm)
        coin_sizes = {
            "quarter": 2.426,    # US Quarter
            "peso": 2.4,         # Philippine Peso
            "nickel": 2.121,     # US Nickel
            "dime": 1.791,       # US Dime
            "penny": 1.955       # US Penny
        }
        
        if coin_type in coin_sizes:
            self.reference_object_size_cm = coin_sizes[coin_type]
            print(f"Using {coin_type} as reference ({self.reference_object_size_cm} cm diameter)")
            # You would need to implement coin detection here
            # For now, this is a placeholder
        else:
            print(f"Unknown coin type: {coin_type}")
    
    def calculate_mango_dimensions(self, detection):
        """
        Calculate real-world dimensions of a mango
        
        Args:
            detection: Detection dictionary from detect_mangoes()
            
        Returns:
            Dictionary with length, width, and area measurements
        """
        if self.pixels_per_cm is None:
            print("Error: System not calibrated. Please calibrate first.")
            return None
        
        x1, y1, x2, y2 = detection['box']
        
        # Calculate dimensions in pixels
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        # Convert to real-world measurements
        width_cm = width_pixels / self.pixels_per_cm
        height_cm = height_pixels / self.pixels_per_cm
        
        # Determine length vs width (length is typically the larger dimension)
        length_cm = max(width_cm, height_cm)
        width_cm = min(width_cm, height_cm)
        
        # Calculate area and perimeter
        area_cm2 = length_cm * width_cm
        perimeter_cm = 2 * (length_cm + width_cm)
        
        # Calculate approximate volume (assuming ellipsoid shape)
        # Volume = (4/3) * π * a * b * c, where c ≈ (a+b)/2 for mango
        a = length_cm / 2
        b = width_cm / 2
        c = (a + b) / 2
        volume_cm3 = (4/3) * math.pi * a * b * c
        
        return {
            'length_cm': round(length_cm, 2),
            'width_cm': round(width_cm, 2),
            'area_cm2': round(area_cm2, 2),
            'perimeter_cm': round(perimeter_cm, 2),
            'volume_cm3': round(volume_cm3, 2),
            'pixels_per_cm': round(self.pixels_per_cm, 2),
            'box_pixels': {
                'width': round(width_pixels, 1),
                'height': round(height_pixels, 1)
            }
        }
    
    def analyze_image(self, image_path, confidence_threshold=0.5):
        """
        Complete analysis of an image with mango measurements
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of mango measurements
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return []
        
        # Detect mangoes
        detections = self.detect_mangoes(image, confidence_threshold)
        
        if not detections:
            print("No mangoes detected in the image")
            return []
        
        # Calculate measurements for each mango
        results = []
        for detection in detections:
            measurements = self.calculate_mango_dimensions(detection)
            if measurements:
                result = {
                    'mango_id': detection['id'],
                    'class': detection['class_name'],
                    'confidence': round(detection['score'], 3),
                    'measurements': measurements,
                    'bounding_box': detection['box'].tolist()
                }
                results.append(result)
        
        return results
    
    def visualize_measurements(self, image_path, save_output=True):
        """
        Visualize mangoes with measurement annotations
        
        Args:
            image_path: Path to input image
            save_output: Whether to save the annotated image
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Get measurements
        results = self.analyze_image(image_path)
        
        if not results:
            print("No mangoes found to measure")
            return
        
        # Draw annotations
        for result in results:
            box = result['bounding_box']
            measurements = result['measurements']
            
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare measurement text
            length = measurements['length_cm']
            width = measurements['width_cm']
            area = measurements['area_cm2']
            
            text_lines = [
                f"ID: {result['mango_id']} ({result['class']})",
                f"L: {length} cm",
                f"W: {width} cm",
                f"Area: {area} cm²"
            ]
            
            # Draw text background and text
            y_offset = y1 - 10
            for i, line in enumerate(text_lines):
                text_y = y_offset - (len(text_lines) - i - 1) * 25
                
                # Text background
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, text_y - text_height - 5), (x1 + text_width, text_y + 5), (0, 255, 0), -1)
                
                # Text
                cv2.putText(image, line, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Display image
        cv2.imshow('Mango Measurements', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save annotated image
        if save_output:
            output_path = image_path.replace('.', '_measured.')
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved as: {output_path}")
    
    def batch_analyze(self, image_folder, output_csv=None):
        """
        Analyze multiple images and optionally save results to CSV
        
        Args:
            image_folder: Folder containing images
            output_csv: Path to save CSV results (optional)
        """
        import os
        import pandas as pd
        
        all_results = []
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_file}")
            
            results = self.analyze_image(image_path)
            
            for result in results:
                row = {
                    'image_file': image_file,
                    'mango_id': result['mango_id'],
                    'class': result['class'],
                    'confidence': result['confidence'],
                    'length_cm': result['measurements']['length_cm'],
                    'width_cm': result['measurements']['width_cm'],
                    'area_cm2': result['measurements']['area_cm2'],
                    'volume_cm3': result['measurements']['volume_cm3']
                }
                all_results.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")
        
        # Print summary statistics
        if not df.empty:
            print("\n=== MEASUREMENT SUMMARY ===")
            print(f"Total mangoes measured: {len(df)}")
            print(f"Average length: {df['length_cm'].mean():.2f} cm")
            print(f"Average width: {df['width_cm'].mean():.2f} cm")
            print(f"Average area: {df['area_cm2'].mean():.2f} cm²")
            print(f"Length range: {df['length_cm'].min():.2f} - {df['length_cm'].max():.2f} cm")
            print(f"Width range: {df['width_cm'].min():.2f} - {df['width_cm'].max():.2f} cm")
        
        return df

# Example usage functions
def calibrate_and_measure_single_image():
    """Example: Calibrate with reference object and measure mangoes"""
    
    # Initialize system
    measurement_system = MangoMeasurementSystem('mango_detection_model.pth')
    
    # Load image with reference object (e.g., ruler, coin, known object)
    image_path = 'img1.png'
    image = cv2.imread(image_path)
    
    # Manual calibration with reference object
    # You need to manually identify the reference object bounding box
    # (980, 435, 1164, 612)
    reference_box = [980, 435, 1164, 612]  # [x1, y1, x2, y2] of reference object
    reference_size_cm = 2.4  # Known size of reference object in cm
    
    # Calibrate
    measurement_system.calibrate_with_reference_object(image, reference_box, reference_size_cm)
    
    # Measure mangoes
    results = measurement_system.analyze_image(image_path)
    
    # Print results
    for result in results:
        print(f"\nMango {result['mango_id']} ({result['class']}):")
        print(f"  Length: {result['measurements']['length_cm']} cm")
        print(f"  Width: {result['measurements']['width_cm']} cm")
        print(f"  Area: {result['measurements']['area_cm2']} cm²")
        print(f"  Confidence: {result['confidence']}")
    
    # Visualize results
    measurement_system.visualize_measurements(image_path)

def interactive_calibration():
    """Interactive calibration helper"""
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['points'].append((x, y))
            if len(param['points']) == 2:
                cv2.destroyAllWindows()
    
    print("Interactive Calibration Tool")
    print("1. Click on two corners of a reference object")
    print("2. Enter the known size of the reference object")
    
    image_path = input("Enter image path: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Could not load image")
        return
    
    # Get reference object coordinates
    param = {'points': []}
    cv2.namedWindow('Select Reference Object')
    cv2.setMouseCallback('Select Reference Object', mouse_callback, param)
    
    print("Click on two opposite corners of the reference object...")
    cv2.imshow('Select Reference Object', image)
    cv2.waitKey(0)
    
    if len(param['points']) != 2:
        print("Need exactly 2 points")
        return
    
    # Calculate reference box
    p1, p2 = param['points']
    reference_box = [min(p1[0], p2[0]), min(p1[1], p2[1]), 
                    max(p1[0], p2[0]), max(p1[1], p2[1])]
    
    reference_size_cm = float(input("Enter the known size of the reference object (in cm): "))
    
    # Initialize and calibrate system
    measurement_system = MangoMeasurementSystem('mango_detection_model.pth')
    measurement_system.calibrate_with_reference_object(image, reference_box, reference_size_cm)
    
    # Measure and visualize
    measurement_system.visualize_measurements(image_path)

def example_use():
    print("Mango Measurement System")
    print("1. Single image measurement")
    print("2. Interactive calibration")
    print("3. Batch processing")
    
    choice = input("Choose option (1-3): ")
    
    if choice == "1":
        calibrate_and_measure_single_image()
    elif choice == "2":
        interactive_calibration()
    elif choice == "3":
        system = MangoMeasurementSystem('mango_detection_model.pth')
        # You'll need to calibrate first before batch processing
        print("Remember to calibrate the system first!")
        folder = input("Enter image folder path: ")
        system.batch_analyze(folder, 'mango_measurements.csv')
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Example usage
    example_use()
    # system = MangoMeasurementSystem('mango_detection_model.pth')    
    # img = 'img1.png'
    # system.auto_calibrate_with_coin(img, "peso")
