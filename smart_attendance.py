import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import pyttsx3 as textSpeech

engine = textSpeech.init()


def resize(img,size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)


path = "student_images"
studentimg = []
studentName = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    studentimg.append(curImg)
    studentName.append(os.path.splitext(cl)[0])
#print(studentName)

def finEncoding(images):
    encoding_list = []
    for img in images:
        img = resize(img,0.5)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        encoding_list.append(encodeimg)
    return encoding_list


def MarkAttendance(name):
    with open('attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H: %M')
            f.writelines(f'\n{name}, {timestr}')
            statement = str("Welcome to class "+name)
            engine.say(statement)
            engine.runAndWait()


EncodeList = finEncoding(studentimg)

vid = cv2.VideoCapture(0)

while True:
    success,frame = vid.read()
    Smaller_frames = cv2.resize(frame,(0,0),None,0.25,0.25)
    frames = cv2.cvtColor(Smaller_frames,cv2.COLOR_BGR2RGB)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)

    cv2.imshow('video',frame)
    cv2.waitKey(1)