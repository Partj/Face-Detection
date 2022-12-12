import cv2 as cv

cas=cv.CascadeClassifier('haar_face.xml')

cam=cv.VideoCapture(0)

while True:
    _,frame=cam.read()
    grey =cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face=cas.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=12, minSize=(30,30), flags=cv.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in face:
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,225), thickness=5)

    cv.imshow('Face Detection',frame)
    if cv.waitKey(1) == ord('0'):
        break

cam.release()
cv.destroyAllWindows()