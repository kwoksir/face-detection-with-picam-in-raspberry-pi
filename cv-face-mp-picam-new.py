import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import mediapipe as mp
import time


camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))
buffer_size = (camera.resolution[0]*camera.resolution[1]*3)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True, resize=(640,480)):
    img = frame.array
    #img_f = cv2.flip(img,0)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw),int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
            #mpDraw.draw_detection(img_f, detection)
    rawCapture.truncate(0)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
camera.close()
