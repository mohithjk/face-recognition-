import numpy as np
import cv2 as cv
import os
harr_cascade=cv.CascadeClassifier('faces.xml')
features=np.load('features.npy', allow_pickle=True)
labels=np.load('labels.npy', allow_pickle=True)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
p=[]
for i in os.listdir(r'C:\Users\JAYANTH\Desktop\opencv\images'):
     p.append(i)
# path=input("enter the path of the image: ")
img=cv.imread(r'C:\Users\JAYANTH\Desktop\opencv\vali\virat.jpeg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)
faces_rect=harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
print(f'the number of faces dected in the image is {len(faces_rect)}')
for (x, y, w, h) in faces_rect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print(f'Label: {p[label]}, Confidence: {confidence}')
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.putText(img, p[label], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
cv.imshow("Detected Faces", img)
cv.waitKey(0)
