import cv2 as cv
import numpy as np

img = cv.imread('PennAir 2024 App Static.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

im2, contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img2 = img

cv.drawContours(img2, im2, 0, (255, 0, 0), 2)
cv.imshow("show contours", img2)
cv.waitKey(0)
cv.destroyAllWindows()
shape = ""
for c in im2:
    approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        shape = "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    elif len(approx) > 6:
        shape = "circle"
    else:
        shape = ""

    cv.drawContours(img, [c], 0, (0, 0, 255), 2)
    cv.putText(img, shape, (approx[0][0][0], approx[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv.imshow("show contours", img)
cv.waitKey(0)
cv.destroyAllWindows()


