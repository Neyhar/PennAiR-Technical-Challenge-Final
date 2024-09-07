import cv2
from ImageDetection import ImageDetection

detector = ImageDetection()
img = cv2.imread('ImagesAndVideos/PennAir 2024 App Static.png')

img = detector.findImage(img)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()