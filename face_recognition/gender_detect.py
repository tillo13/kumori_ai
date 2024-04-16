import cv2
import os

# Model file paths for face and gender detection were already in your code
faceProto = "gender_detect/opencv_face_detector.pbtxt"
faceModel = "gender_detect/opencv_face_detector_uint8.pb"
genderProto = "gender_detect/gender_deploy.prototxt"
genderModel = "gender_detect/gender_net.caffemodel"

# Add model file paths for age detection
ageProto = "gender_detect/age_deploy.prototxt"
ageModel = "gender_detect/age_net.caffemodel"

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Load age detection model

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


# Function for highlighting faces
def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes

# Detect gender function

def detect_gender(frame):
    faceBoxes = highlightFace(faceNet, frame)
    numberOfFacesDetected = len(faceBoxes)

    if not faceBoxes:
        return ("man_suit.png", "None", 0, "0-0", 0, numberOfFacesDetected)  # Default, if no faces

    # Considering only the first detected face for simplicity
    faceBox = faceBoxes[0]
    
    gender_confidence = 0
    age_confidence = 0
    
    face = frame[max(0,faceBox[1]-20):min(faceBox[3]+20,frame.shape[0]-1), max(0,faceBox[0]-20):min(faceBox[2]+20,frame.shape[1]-1)]
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    gender_confidence = genderPreds[0].max()
    
    # Age Detection (Similar to gender detection)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    age_confidence = agePreds[0].max()

    chosen_suit = "woman_suit.png" if gender == 'Female' else "man_suit.png"
    
    return (chosen_suit, gender, gender_confidence, age, age_confidence, numberOfFacesDetected)