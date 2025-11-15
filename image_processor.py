import cv2
import numpy as np

def extract_geometric_features(image_path: str):
    """
    Calculates geometric features from a single-layer image.

    The function reads an image, finds the largest contour, and calculates
    its area, perimeter, and hydraulic diameter.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the calculated features:
              'area', 'perimeter', and 'hydraulic_diameter'.
              Returns None if no contours are found.
    """
    try:
        # 1. Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be read. Check the file path.")

        # 2. Threshold the image to get a binary mask
        # We assume the object is white (or lighter) on a black (or darker) background.
        # cv2.THRESH_OTSU automatically determines the optimal threshold value.
        _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Find contours
        # cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
        # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Warning: No contours found in the image.")
            return {
                'area': 0,
                'perimeter': 0,
                'hydraulic_diameter': 0
            }

        # 4. Assume the largest contour is the object of interest
        largest_contour = max(contours, key=cv2.contourArea)

        # 5. Calculate features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Avoid division by zero
        if perimeter == 0:
            hydraulic_diameter = 0
        else:
            # Hydraulic Diameter = 4 * Area / Perimeter
            hydraulic_diameter = 4 * area / perimeter
            
        return {
            'area': area,
            'perimeter': perimeter,
            'hydraulic_diameter': hydraulic_diameter
        }

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None

if __name__ == '__main__':
    # --- Example of how to use the function ---
    # Create a dummy image for testing purposes.
    # A white 200x100 rectangle on a black background.
    test_image_path = 'test_image.png'
    dummy_image = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(dummy_image, (50, 75), (250, 175), 255, -1) # x1,y1, x2,y2
    cv2.imwrite(test_image_path, dummy_image)

    print(f"Created a test image: {test_image_path}")

    # Extract features from the dummy image
    features = extract_geometric_features(test_image_path)

    if features:
        print("\n--- Extracted Geometric Features ---")
        # Expected area = 200 * 100 = 20000
        # Expected perimeter = 2 * (200 + 100) = 600
        # Expected hydraulic diameter = 4 * 20000 / 600 = 133.33
        print(f"Area: {features['area']:.2f} (Expected: 20000)")
        print(f"Perimeter: {features['perimeter']:.2f} (Expected: 600)")
        print(f"Hydraulic Diameter: {features['hydraulic_diameter']:.2f} (Expected: 133.33)")
        print("------------------------------------")

    # Clean up the dummy image
    import os
    os.remove(test_image_path)
    print(f"Removed test image: {test_image_path}")
