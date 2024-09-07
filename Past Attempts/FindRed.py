import cv2
import numpy as np

# Load the image
image = cv2.imread('PennAir 2024 App Static.png')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper range for dark red color in HSV
lower_red = np.array([0, 50, 50])    # Lower bound of hue, saturation, and value
upper_red = np.array([10, 255, 255])  # Upper bound of hue, saturation, and value

# Create a mask for the red color
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# Also check the upper range for red hues
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine the masks to cover the full red hue range
mask = mask1 + mask2

# Perform morphological operations to remove noise (optional)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours to detect rectangles
for contour in contours:
    # Approximate the contour to a polygon and check if it has 4 vertices
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    
    # Check if the approximated contour has 4 points (indicating a rectangle)
    if len(approx) == 4:
        # Compute the area of the contour to filter out small noise
        area = cv2.contourArea(contour)
        if area > 100:  # Threshold for minimum area (adjust as needed)
            # Draw the contour on the image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

            # You can also get the bounding box if needed
            x, y, w, h = cv2.boundingRect(approx)
            print(f"Rectangle found at (x: {x}, y: {y}), width: {w}, height: {h}")

# Display the original image with detected rectangles
cv2.imshow('Detected Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()