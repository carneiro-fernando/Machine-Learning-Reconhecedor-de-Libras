# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:01:48 2022

@author: Fernando H.C. Carneiro
"""

# Importando as bibliotecas necessárias
from keras.models import model_from_json
import operator
import cv2

# Carregando o modelo
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

# Carregando os pesos no modelo
loaded_model.load_weights("model-bw.h5")

# Instanciando captura de video
cap = cv2.VideoCapture(0)

# Criando dicionário com as letras U-N-I-V-E-S-P
categories = {0: 'E', 1: 'I', 2: 'N', 3: 'P', 4: 'S', 5: 'U', 6:'V'}

# Enquanto houver captura de vídeo, faça:
while True:
    _, frame = cap.read()
    
    # Transladando a imagem horizontalmente (espelho)
    frame = cv2.flip(frame, 1)
    
    # Declarando variáveis de coordenada da zona de interesse
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    
    # Desenhando a zona de interesse (retângulo verde)
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0) ,1)
    
    # Instanciando zona de interesse
    roi = frame[y1:y2, x1:x2]
    
    # Manipulando a imagem dentro da zona de interesse
    roi = cv2.resize(roi,(64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Exibindo a imagem capturada
    cv2.imshow("Captura", roi)
    
    # Previsão do resultado com base no modelo neural
    result = loaded_model.predict(roi.reshape(1, 64, 64, 1))
    prediction = {'E': result[0][0], 
                  'I': result[0][1], 
                  'N': result[0][2],
                  'P': result[0][3],
                  'S': result[0][4],
                  'U': result[0][5],
                  'V': result[0][6]}
    
    # Identificando qual a previsão mais próxima
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Exibindo a previsão
    cv2.putText(frame, prediction[0][0], (100, 120), cv2.FONT_HERSHEY_PLAIN, 3, (14,14,14), 3)    
    cv2.imshow("Frame", frame)
    
    # Interromper com a tecla ESC
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

# Finaliza captura de vídeo e fecha as janelas    
cap.release()
cv2.destroyAllWindows()