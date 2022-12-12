import cv2
import os

# Train or test 
mode = 'treino'
directory = 'dataset/'+mode+'/'

    
if not os.path.exists("dataset"):
    os.makedirs("dataset")
    os.makedirs("dataset/treino")
    os.makedirs("dataset/teste")
    
    os.makedirs("dataset/treino/U")
    os.makedirs("dataset/treino/N")
    os.makedirs("dataset/treino/I")
    os.makedirs("dataset/treino/V")
    os.makedirs("dataset/treino/E")
    os.makedirs("dataset/treino/S")
    os.makedirs("dataset/treino/P")
    
    os.makedirs("dataset/teste/U")
    os.makedirs("dataset/teste/N")
    os.makedirs("dataset/teste/I")
    os.makedirs("dataset/teste/V")
    os.makedirs("dataset/teste/E")
    os.makedirs("dataset/teste/S")
    os.makedirs("dataset/teste/P")
    
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'E': len(os.listdir(directory+"E")),
             'I': len(os.listdir(directory+"I")),
             'N': len(os.listdir(directory+"N")),
             'P': len(os.listdir(directory+"P")),
             'S': len(os.listdir(directory+"S")),
             'U': len(os.listdir(directory+"U")),
             'V': len(os.listdir(directory+"V"))}
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "U : "+str(count['U']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "N : "+str(count['N']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "I : "+str(count['I']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "V : "+str(count['V']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "E : "+str(count['E']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "S : "+str(count['S']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    cv2.putText(frame, "P : "+str(count['P']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
 
    cv2.imshow(mode, frame)
    
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)   
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['U'])+'.png', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.png', roi)        
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.png', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['V'])+'.png', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.png', roi)  
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['S'])+'.png', roi)
    if interrupt & 0xFF == ord('p'):
       cv2.imwrite(directory+'P/'+str(count['P'])+'.png', roi)
    
cap.release()
cv2.destroyAllWindows()






 #_, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)
 
 
 
 #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
 #roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel,iterations=1)
 #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
 #roi = cv2.GaussianBlur(roi,(3,3),0)#
 #_, roi = cv2.threshold(roi, 255,255,cv2.THRESH_TOZERO)
 #roi= cv2.medianBlur(roi, 3)