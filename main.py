from model import EmotionDetector
import cv2
import imutils
import numpy as np

emotion_detector = EmotionDetector()
face_detection = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    # reading the frame
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    for fX, fY, fW, fH in faces:
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the Xception
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))

        roi = roi.astype('float') / 255.0
        roi = roi - 0.5
        roi = roi * 2.0

        roi = np.expand_dims(roi, axis=-1)

        pred = emotion_detector.predict([roi])
        label = EMOTIONS[pred[0]]

        # construct the label text
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    cv2.imshow('Emotion recognizer', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

