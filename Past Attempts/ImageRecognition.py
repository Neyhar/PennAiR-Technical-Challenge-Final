import cv2 as cv
import numpy as np

image = cv.imread('PennAir 2024 App Static.png')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (3,3), 0)

_, thresh = cv.threshold(blur, 104, 255, cv.THRESH_BINARY)

_, invthresh = cv.threshold(blur, 104, 255, cv.THRESH_BINARY_INV)

cv.imshow("thresh", thresh)
cv.waitKey(0)
cv.destroyAllWindows()


cv.imshow("thresh", invthresh)
cv.waitKey(0)
cv.destroyAllWindows()


contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

edges = cv.Canny(image=blur, threshold1=100, threshold2=200)
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)
cv.destroyAllWindows()
for contour in contours:
    epsilon = 0.001 * cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    if len(approx) == 3:
        shape_name = "Triangle"
    elif len(approx) == 4:
        # Check if it's a rectangle or trapezoid
        (x, y, w, h) = cv.boundingRect(approx)
        aspectRatio = w / float(h)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape_name = "Rectangle"
        else:
            shape_name = "Trapezoid"
    elif len(approx) == 5:
        shape_name = "Pentagon"
    elif len(approx) == 6:
        shape_name = "Hexagon"
    else:
        shape_name = "Circle"
    
    cv.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Step 7: Show the Result
cv.imshow('Detected Shapes', image)
cv.waitKey(0)
cv.destroyAllWindows()












