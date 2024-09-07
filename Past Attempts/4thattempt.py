import cv2
 
# Read the original image
img = cv2.imread('PennAir 2024 App Static.png') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    epsilon = 0.001 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 3:
        shape_name = "Triangle"
    elif len(approx) == 4:
        # Check if it's a rectangle or trapezoid
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape_name = "Rectangle"
        else:
            shape_name = "Trapezoid"
    elif len(approx) == 5:
        shape_name = "Pentagon"
    elif len(approx) == 6:
        shape_name = "Hexagon"
    elif len(approx) > 7:
        shape_name = "Circle"
    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

cv2.imshow("image", img)
cv2.destroyAllWindows()