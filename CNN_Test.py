import cv2
import pickle
import numpy as np
# from tensorflow.keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential





cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

pickle_in =open("pickle_file/model_trained.p","rb")
model = pickle.load(pickle_in)


def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success,imagOriginal=cap.read()
    img=np.array(imagOriginal)
    img=cv2.resize(img,(32,32))
    img=preProcessing(img)
    cv2.imshow("proceessed Image",img)
    img=img.reshape(1,32,32,1)
    #predicting.....
    classIndex=int(model.predict_classes(img))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
