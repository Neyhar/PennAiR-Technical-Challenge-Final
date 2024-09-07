import cv2 as cv
from ImageDetection import ImageDetection

path = 'ImagesAndVideos/PennAir 2024 App Dynamic.mp4'
cap = cv.VideoCapture(path)
detector = ImageDetection()

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    frame1 = detector.findImage(frame)

    cv.imshow('Processed Frame', frame1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
