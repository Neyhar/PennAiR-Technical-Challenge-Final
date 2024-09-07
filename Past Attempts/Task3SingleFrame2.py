import cv2
import numpy as np

# Load the image
image = cv2.imread('ImagesAndVideos/Gradient Hard.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through each contour to identify the shape
for contour in contours:
    # Approximate the contour to reduce the number of points
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Determine the number of vertices of the approximated contour
    vertices = len(approx)
    
    # Calculate the center of the contour for text placement
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    # Identify the shape based on the number of vertices
    if vertices == 3:
        shape_name = "Triangle"
    elif vertices == 4:
        # Use aspect ratio to differentiate between square and rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        shape_name = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif vertices == 5:
        shape_name = "Pentagon"
    elif vertices == 6:
        shape_name = "Hexagon"
    else:
        # Use circularity to detect circles
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        shape_name = "Circle" if 0.7 < circularity < 1.3 else "Ellipse"

    # Draw the contour and label the shape
    area = cv2.contourArea(contour)
    if area > 500:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    

# Show the image with detected shapes
cv2.imshow("Detected Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()