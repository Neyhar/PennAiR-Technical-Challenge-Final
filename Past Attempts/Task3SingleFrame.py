import cv2
from ImageDetection import ImageDetection

detector = ImageDetection()
vid = cv2.VideoCapture('ImagesAndVideos/PennAir 2024 App Dynamic Hard.mp4')
_, img = vid.read()
noise = cv2.imread('ImagesAndVideos/noise2.png')

noise2 = cv2.GaussianBlur(noise, [5,5], 7)

img2 = detector.subtract_frames(img, noise2)
cv2.imshow('find', img2)
cv2.waitKey(0)

img = detector.findBinary(img2, img)
cv2.imshow('find', img)
cv2.waitKey(0)

cv2.destroyAllWindows()