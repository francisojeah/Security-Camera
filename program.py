import cv2 as cv
import datetime
import time

cam = cv.VideoCapture(0)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
bodyCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')

detection = False
detectionTime =None
startedTimer = False 
recordTimeAfterDetection = 3

currentTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

frameSize = (int(cam.get(3)), int(cam.get(4)))
four_cc = cv.VideoWriter_fourcc(*'mp4v')

while True:
    success, frame = cam.read()

    greyConvert = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    foundFaces = faceCascade.detectMultiScale(greyConvert , 1.3, 5)
    foundBodies = bodyCascade.detectMultiScale(greyConvert, 1.3, 5)

    if len(foundFaces) + len(foundBodies):
        if detection:
            startedTimer = False

        else:
            detection = True
            output = cv.VideoWriter(f"{currentTime}.mp4", four_cc, 20, frameSize)
            print('Recording started')

    elif detection:
        if startedTimer:
            if time.time() - detectionTime >= recordTimeAfterDetection:
                detection = False
                startedTimer = False
                output.release()
                print('Stop Recording')

            else:
                startedTimer = True
                detectionTime= time.time()

    if detection:        
        output.write(frame)

    for (x, y, width, height) in foundFaces:
        cv.rectangle(frame, (x,y), (x+width, y+height), (255,0,0), 2)
        
    cv.imshow('video',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
output.release()
cam.release()
cv.destroyAllWindows()