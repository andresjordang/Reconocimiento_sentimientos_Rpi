from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

num_classes=5               # 5 clases
img_rows,img_cols=48,48     # Resolucion de imagenes de entrada
batch_size=32               # Lotes de imagenes
epochs=18                   # Repeticiones del entrenamiento

train_data_dir='C:/Users/jorda/Desktop/Diseno_electronico/Parte 1/Proyecto_Rpi/train'
validation_data_dir='C:/Users/jorda/Desktop/Diseno_electronico/Parte 1/Proyecto_Rpi/validation'

# Opciones para la expansion de dataset mediante la transformacion de las imagenes disponibles
# El escalado tambien se configura en este paso

train_datagen = ImageDataGenerator(
					rescale=1./255,            # Escalado
					rotation_range=30,         # Radio de rotacion
					zoom_range=0.3,            # Rango de zoom
					width_shift_range=0.4,     # Cambio de ancho
					height_shift_range=0.4,    # Cambio de alto 
					horizontal_flip=True,      # Inversion horizontal
					fill_mode='nearest')       # Rellena los posibles huecos de las imagenes generadas 

validation_datagen = ImageDataGenerator(rescale=1./255) # Los datos de validacion solo se reescalan

# Carga del dataset con la opciones de expasion, y otras modificaciones

train_generator = train_datagen.flow_from_directory(
					train_data_dir,                    # Ruta del dataset de entrenamiento
					color_mode='grayscale',            # En escala de grises (mas facil para red neuronal)
					target_size=(img_rows,img_cols),   # Con ancho y alto de 48 pixeles
					batch_size=batch_size,             # Lotes de muestas de actualizacion de pesos
					class_mode='categorical',          # Imagenes categorizadas en 5 clases
					shuffle=True)                      # Mezcla conjunto de datos

validation_generator = validation_datagen.flow_from_directory(
    					validation_data_dir,           # Ruta del dataset de validacion
						color_mode='grayscale',
						target_size=(img_rows,img_cols),
						batch_size=batch_size,
						class_mode='categorical',
						shuffle=True)

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

print(model.summary())

# Modelo se complia y entrena, con un numero de repeticiones(epoch) optimizado

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

history=model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                batch_size=batch_size,
                verbose=1)

# Guardado del modelo, para luago cargarlo en la Raspberry

model.save_weights("Pesos_modelo.h5")

# Genera graficas del entrenamiento

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left')
plt.ylim([0.3, 1])


plt.figure(2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss del modelo')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left')
