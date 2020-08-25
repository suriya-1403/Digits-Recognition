import cv2
import pickle
import numpy as np
from keras.models import load_model
# from tensorflow.keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential



model = load_model('trained_cnn_model.h5')
#####################################
#invoking camera
#####################################

cap=cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)

def preProcessing(img):
 img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 img = cv2.equalizeHist(img)
 img = img/255 #normalizing
 return img

while True:
 success,imagOriginal=cap.read()
 img=np.asarray(imagOriginal)
 img=cv2.resize(img,(32,32))
 img=preProcessing(img)
 cv2.imshow("proceessed Image",img)
 img=img.reshape(1,32,32,1)
 #predicting.....
 classIndex=int(model.predict_classes(img))
 #print(classIndex)
 prediction=model.predict(img)
 probability_value=np.argmax(prediction,axis=-1)
 print(classIndex,probability_value)

 if probability_value>0.65:
   cv2.putText(imagOriginal,str(classIndex) + "   "+str(probability_value),
                     (50,50),cv2.FONT_HERSHEY_COMPLEX,
                     1,(0,0,255),1)
 
 cv2.imshow("Digits Recognition",imagOriginal)
 if cv2.waitKey(1) & 0xFF == ord('q'):
     break
