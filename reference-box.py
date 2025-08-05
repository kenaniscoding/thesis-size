import cv2
import numpy as np

class ReferenceBoxSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.clone = None
        self.reference_box = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing the reference box"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update the box while dragging
            if self.drawing:
                self.end_point = (x, y)
                # Create a copy of the original image to draw on
                temp_image = self.clone.copy()
                cv2.rectangle(temp_image, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow("Select Reference Box", temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate the reference box coordinates
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Ensure coordinates are in the correct order (top-left to bottom-right)
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            self.reference_box = (left, top, right, bottom)
            
            # Draw the final box
            cv2.rectangle(self.image, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow("Select Reference Box", self.image)
            
            # Print the reference box coordinates
            print(f"Reference Box Coordinates:")
            print(f"Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")
            print(f"Width: {right - left}, Height: {bottom - top}")
            print(f"Reference Box (x, y, w, h): ({left}, {top}, {right - left}, {bottom - top})")
            print(f"Reference Box (x1, y1, x2, y2): ({left}, {top}, {right}, {bottom})")
            
    def select_reference_box(self):
        """Main function to load image and handle reference box selection"""
        try:
            # Load the image
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                print(f"Error: Could not load image '{self.image_path}'")
                print("Please make sure the image file exists and is in a supported format.")
                return None
                
            # Create a copy for drawing operations
            self.clone = self.image.copy()
            
            # Create window and set mouse callback
            cv2.namedWindow("Select Reference Box", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Select Reference Box", self.mouse_callback)
            
            # Display instructions
            print("Instructions:")
            print("1. Click and drag to select a reference box on the image")
            print("2. Press 'r' to reset and select a new box")
            print("3. Press 'q' or ESC to quit")
            print("4. Press 's' to save the current selection")
            
            cv2.imshow("Select Reference Box", self.image)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('r'):  # Reset
                    self.image = self.clone.copy()
                    self.reference_box = None
                    self.start_point = None
                    self.end_point = None
                    cv2.imshow("Select Reference Box", self.image)
                    print("Selection reset. Click and drag to select a new reference box.")
                elif key == ord('s'):  # Save/print current selection
                    if self.reference_box:
                        left, top, right, bottom = self.reference_box
                        print("\n--- SAVED REFERENCE BOX ---")
                        print(f"Reference Box Coordinates:")
                        print(f"Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")
                        print(f"Width: {right - left}, Height: {bottom - top}")
                        print(f"Reference Box (x, y, w, h): ({left}, {top}, {right - left}, {bottom - top})")
                        print(f"Reference Box (x1, y1, x2, y2): ({left}, {top}, {right}, {bottom})")
                        print("--- END REFERENCE BOX ---\n")
                    else:
                        print("No reference box selected yet. Please click and drag to select one.")
            
            cv2.destroyAllWindows()
            return self.reference_box
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def main():
    # Initialize the reference box selector
    selector = ReferenceBoxSelector("img1.png")
    
    # Select the reference box
    reference_box = selector.select_reference_box()
    
    if reference_box:
        print(f"\nFinal Reference Box: {reference_box}")
    else:
        print("No reference box was selected.")

if __name__ == "__main__":
    main()
