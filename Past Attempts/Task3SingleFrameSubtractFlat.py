import cv2 as cv
from ImageDetection import ImageDetection

detector = ImageDetection()

flat = cv.imread('ImagesAndVideos/whitescreen.png')
img = cv.imread('ImagesAndVideos/Gradient Hard.png')

frame1 = cv.subtract(flat, img)

cv.imshow('Processed Frame', frame1)
cv.waitKey(0)

img = detector.findBlueAndTanIn3D(img, frame1)

cv.imshow('Processed Frame', img)
cv.waitKey(0)
cv.destroyAllWindows()