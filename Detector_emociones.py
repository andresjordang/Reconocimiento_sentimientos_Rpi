from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from time import sleep
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')

camera = PiCamera()
camera.resolution = (480, 480)
camera.framerate = 30
camera.rotation = 180

# Definicion de la red neuronal

model = Sequential()

# Todas las capas cuentan con BatchNormalization y Dropout (tecnicas para evitar el sobreajuste)

#Bloque-1: 2 capas convolucionales de 32 filtros y unz capa de pooling

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))                     

#Bloque-2: 2 capas convolucionales de 64 filtros y una capa de pooling

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Bloque-2: 2 capas convolucionales de 128 filtros y una capa de pooling

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Bloque-2: 2 capas convolucionales de 256 filtros y una capa de pooling

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Bloque-5: Capa de flatten y capa densa de 64 neuronas
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Bloque-6: Capa densa de 64 neuronas

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Bloque-7: Capa densa de 5 neuronas, la salida

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

model.load_weights('Pesos_modelo.h5')

print("Introduzca duranción de la deteccion: ")
duracion=input()
resultado=[0 for i in duracion]

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
                    resultado[i]=j
                    
            emociones=['Angry','Happy','Neutral','Sad','Surprise']
            emocion=emociones[predic.argmax()]
            posicion=(x,y)
            cv2.putText(image,emocion,posicion,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imwrite('resultado.jpg',image)
            
        else:
            resultado[i]=-1
            cv2.imwrite('resultado.jpg',image)
camera.close()