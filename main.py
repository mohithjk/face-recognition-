import os
import cv2 as cv
import numpy as np
p=[]
for i in os.listdir(r'C:\Users\JAYANTH\Desktop\opencv\images'):
     p.append(i)
print(p)
DIR=r'C:\Users\JAYANTH\Desktop\opencv\images'
harr_cascade=cv.CascadeClassifier('faces.xml')
features=[]
labels=[]
def train_model():
     for persons in p:
          path=os.path.join(DIR, persons)
          # in this step proper directory path name is created for each person
          label=p.index(persons)
          # this will give the index of the person in the list p
          for img in os.listdir(path):
             img_path=os.path.join(path,img)
             img_array=cv.imread(img_path)
             gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
             faces_rect=harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
             for (x,y,w,h) in faces_rect:
                face_roi=gray[y:y+h, x:x+w]# Extracts the region of interest (ROI) from the grayscale image, i.e., the face area.
                features.append(face_roi)
                labels.append(label)
train_model()
print("training is done")
print(f'Number of faces found: {len(features)}')
features=np.array(features, dtype='object')
labels=np.array(labels, dtype='int32')  
face_recognizer=cv.face.LBPHFaceRecognizer_create()
# LBPH stands for Local Binary Patterns Histograms
#here we r instintiating the face recoginzer
face_recognizer.train(features, labels)
face_recognizer.save('face_recognizer.yml')
# this will save the trained face recognizer to a file named 'face_recognizer.yml
# this will train the face recognizer with the features and labels
np.save('features.npy', features)
np.save('labels.npy', labels)   

