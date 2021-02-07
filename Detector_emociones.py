from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from keras.models import load_model
from time import sleep
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
clasificador = load_model('Pesos_modelo.h5')

camera = PiCamera()
camera.resolution = (480, 480)
camera.framerate = 30
camera.rotation = 180

print("Introduzca duranción de la deteccion: ")
duracion=input()
resultado[duracion]=[]

for i in duracion:
    image = np.empty((480, 480, 3), dtype=np.uint8)
    camera.capture(Img, 'rgb')
    
    #Detección
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im_gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        im_face=im_gray[y:y+h,x:x+w]
        im_face=cv2.resize(im_face,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([im_face])!=0:
            img=im_face.astype('float')/255.0
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)

            predic=clasificador.predict(img)[0]
            
            for j in range(5):
                if predic(j)==1:
                    resultado(i)=j
                    
            emociones=['Angry','Happy','Neutral','Sad','Surprise']
            emocion=emociones[predic.argmax()]
            posicion=(x,y)
            cv2.putText(image,emocion,posicion,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imwrite('resultado.jpg',image)
            
        else:
            resultado(i)=-1
            cv2.imwrite('resultado.jpg',image)
camera.close()