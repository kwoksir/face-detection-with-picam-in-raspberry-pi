import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320,240))
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

time.sleep(1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    faces = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img,"face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 1)

    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break
camera.close()
cv2.destroyAllWindows()