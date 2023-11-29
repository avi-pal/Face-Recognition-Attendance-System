#importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec


#function
def resize(img,size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)


#img declaration
avirup1 = face_rec.load_image_file('sample_images/avirup1.jpg')
avirup1 = cv2.cvtColor (avirup1, cv2.COLOR_BGR2RGB)
avirup1 = resize(avirup1,0.50)
avirup2 = face_rec.load_image_file('sample_images/avirup2.jpg')
avirup2 = cv2.cvtColor (avirup2, cv2.COLOR_BGR2RGB)
avirup2 = resize(avirup2,0.50)





#find face location
faceLocation_avirup1 = face_rec.face_locations(avirup1)[0]
encode_avirup1 = face_rec.face_encodings(avirup1)[0]
cv2.rectangle(avirup1,(faceLocation_avirup1[3],faceLocation_avirup1[0]),(faceLocation_avirup1[1],faceLocation_avirup1[2]),(255,0,255),3)

faceLocation_avirup2 = face_rec.face_locations(avirup2)[0]
encode_avirup2 = face_rec.face_encodings(avirup2)[0]
cv2.rectangle(avirup2,(faceLocation_avirup2[3],faceLocation_avirup2[0]),(faceLocation_avirup2[1],faceLocation_avirup2[2]),(255,0,255),3)

print(encode_avirup1)
print(encode_avirup2)

results = face_rec.compare_faces([encode_avirup1],encode_avirup2)
print(results)
cv2.putText(avirup2,f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)










cv2.imshow('main_img', avirup1)
cv2.imshow('test_img', avirup2)
cv2.waitKey()
cv2.destroyAllWindows()