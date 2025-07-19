import cv2 as cv 
img=cv.imread('virat.jpeg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)
cv.waitKey(0)
harr_cascade=cv.CascadeClassifier('faces.xml')
#thrigh this code we r implementing all the code that is in the haar cascade file to this
faces_rect=harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
#through this code we r detecting the faces in the image
#faces_rect basically stores the list of the rectangle coordinates
#  of the face found in the image 
print(f'the number of faces dected in the image is {len(faces_rect)}')
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (255,255,0), thickness=2)
cv.imshow("Detected Faces", img)
cv.waitKey(0)
# this is not something advanced but it is a simple way to detect faces in an image