import cv2
import numpy as np

def calculate_pixels_per_cm(x1, y1, x2, y2, reference_length_cm, reference_width_cm):
    # Calculate pixel dimensions
    length_pixels = abs(x2 - x1)
    width_pixels = abs(y2 - y1)
    
    # Calculate pixels per cm for each dimension
    pixels_per_cm_length = length_pixels / reference_length_cm
    pixels_per_cm_width = width_pixels / reference_width_cm
    
    # Return average
    return (pixels_per_cm_length + pixels_per_cm_width) / 2

# Example usage:
if __name__ == "__main__":
    # Your specific case
    x1, y1, x2, y2 = 36, 8, 1875, 779
    pixels_per_cm = calculate_pixels_per_cm(x1, y1, x2, y2, 
                                          reference_length_cm=2.45, 
                                          reference_width_cm=11.0)
    
    print(f"Pixels per cm: {pixels_per_cm:.2f}")
    
    # Now you can use this to measure other objects:
    object_length_pixels = 300
    object_length_cm = object_length_pixels / pixels_per_cm
    print(f"Object measuring {object_length_pixels} pixels = {object_length_cm:.2f} cm") 
