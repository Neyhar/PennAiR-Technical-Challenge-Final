import cv2 as cv
from ImageDetection import ImageDetection

detector = ImageDetection()

path = 'ImagesAndVideos/PennAir 2024 App Dynamic Hard.mp4'
cap = cv.VideoCapture(path)
_, img = cap.read()
detector = ImageDetection()
noise = cv.imread('ImagesAndVideos/noise.png')
flat = cv.imread('ImagesAndVideos/whitescreen.png')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    frame1 = cv.subtract(flat, frame)

    frame = detector.findContoursSelf2(frame, frame1)
    frame = detector.findBlueAndTan(frame, frame1)

    cv.imshow('Processed Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()