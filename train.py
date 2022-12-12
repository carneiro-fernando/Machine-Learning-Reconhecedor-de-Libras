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
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

## Construindo a rede neural

# Inicialização
model = Sequential()

# Primeira camada de convolução
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

# Segunda camada de convolução
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Terceira camada de convolução
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Quarta camada de convolução
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convertendo as matrizes em arrays
model.add(Flatten())

# Camada totalmente conectada
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=7, activation='softmax'))

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## Preparando os dados e parametros

# Constroi um conjunto de dados com arquivos de imagem (treino)
train_datagen = ImageDataGenerator(rescale=1./255)
       

# Constroi um conjunto de dados com arquivos de imagem (teste)
test_datagen = ImageDataGenerator(rescale=1./255)

# Definindo localização e formato das imagens para treino
training_set = train_datagen.flow_from_directory('dataset/treino',
                                                 target_size=(64, 64),
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

# Definindo localização e formato das imagens para teste
test_set = test_datagen.flow_from_directory('dataset/teste',
                                            target_size=(64, 64),
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Definindo os parametros de aprendizagem
history = model.fit(
        training_set,
        epochs=35,
        validation_data=test_set)

#Gráficos de avaliação de resultado
import matplotlib.pyplot as plt

# Renderização de gráfico de acuracia
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

# Renderização de gráfico de perda
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()
    
# Salvando o modelo
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# Salvando os pesos    
model.save_weights('model.h5')