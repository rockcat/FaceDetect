import cv2
import sys
import numpy as np

MINI_SIZE = 200

# Get user supplied values
def getFaceCascade():
    """ Gets the path to the cascade file """
    cascpath = "C:/Users/pauln/Documents/workspace/FaceDetect/haarcascade_frontalface_default.xml"
    if len(sys.argv) > 1:
        cascpath = sys.argv[1]
    # Create the haar cascade
    return cv2.CascadeClassifier(cascpath)

def processImage(cap, cascade, output):
    """ Process a single frame for faces"""
    # Read the image
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
 
    # Draw a rectangle around the faces
    index = 0
    scaled = np.zeros((MINI_SIZE,MINI_SIZE,3), np.uint8)
    for (xpos, ypos, width, height) in faces:
        face = image[ypos:ypos+height,xpos:xpos+width]
        cv2.rectangle(image, (xpos, ypos), (xpos+width, ypos+height), (0, 255, 0), 2)
        msg = "Found : " + str(len(faces)) 
        cv2.putText(image, msg, (xpos, ypos), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        if (face.shape[0] > 0) and (face.shape[1] > 0):
            ratio = face.shape[0] / face.shape[1]

            dim = (MINI_SIZE, int(MINI_SIZE * ratio))
            scaled = cv2.resize(face, (MINI_SIZE,MINI_SIZE))
            xoffs = index * MINI_SIZE
            yoffs = image.shape[0]
            output[yoffs:yoffs+MINI_SIZE, xoffs:xoffs+MINI_SIZE] = scaled
            output[0:yoffs, 0:image.shape[1]] = image
            index = index + 1
    cv2.imshow("Face Detector", output)

def Main():
    CAP = cv2.VideoCapture(0)
    vidWidth = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outputImage = np.zeros((vidHeight + MINI_SIZE,vidWidth,3), np.uint8)
    cascade = getFaceCascade()

    while True:
        processImage(CAP, cascade, outputImage)
        ch = cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()

Main()