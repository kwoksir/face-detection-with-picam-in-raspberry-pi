import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import mediapipe as mp
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=(640,480))

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

buffer_size = (camera.resolution[0]*camera.resolution[1]*3)

time.sleep(1)
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True, resize=(640,480)):
        img = frame.array
        results = face_detection.process(img)
        if results.detections:
            for id, detection in enumerate(results.detections):
                #mp_draw.draw_detection(image, detection)
                #print(id, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                cv2.rectangle(img, boundBox, (0, 255, 0), 3)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        rawCapture.truncate(0)
        cv2.imshow('Face Detection', img)
        if cv2.waitKey(1) == ord('q'):
            break
camera.close()
cv2.destroyAllWindows()
