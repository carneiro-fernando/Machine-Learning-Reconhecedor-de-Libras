# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:15:08 2022

@author: Fernando H.C. Carneiro
"""

# Importando as bibliotecas necessárias
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os

## Construindo a rede neural

# Inicialização
classifier = Sequential()

# Primeira camada de convolução
classifier.add(Convolution2D(32, (4, 4), input_shape=(64, 64, 1), activation='relu'))
# Primeira camada de pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda camada de convolução
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# Segunda camada de pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Terceira camada de convolução
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# Terceira camada de pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Convertendo as matrizes em arrays
classifier.add(Flatten())

# Camada totalmente conectada
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=7, activation='softmax'))

# Compilando o modelo
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## Preparando os dados e parametros

# Constroi um conjunto de dados com arquivos de imagem (treino)
train_datagen = ImageDataGenerator(     
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

# Constroi um conjunto de dados com arquivos de imagem (teste)
test_datagen = ImageDataGenerator(rescale=1./255)

# Definindo localização e formato das imagens para treino
training_set = train_datagen.flow_from_directory('dataset/treino',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

# Definindo localização e formato das imagens para teste
test_set = test_datagen.flow_from_directory('dataset/teste',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Definindo os parametros de aprendizagem
classifier.fit(
        training_set,
        steps_per_epoch=len(os.listdir('dataset/treino/E')), #Função p usar todas as imagens na pasta
        epochs=3000,
        validation_data=test_set,
        validation_steps=len(os.listdir('dataset/teste/E')))

# Salvando o modelo
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
    
# Salvando os pesos    
classifier.save_weights('model-bw.h5')

