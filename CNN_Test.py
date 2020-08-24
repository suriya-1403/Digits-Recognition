import numpy as np
import cv2
import pickle

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL
pickle_in = open("pickle_file/model_trained_10.p","rb")
model = pickle.load(pickle_in)

#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# import cv2
# import pickle
# import numpy as np
# # from tensorflow.keras.models import Sequential
# # from keras.preprocessing.image import ImageDataGenerator
# # from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
#
#
#
#
#
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
#
# pickle_in =open("pickle_file/model_trained_10.p","rb")
# model = pickle.load(pickle_in)
#
#
# def preProcessing(img):
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img/255
#     return img
#
# while True:
#     success,imagOriginal=cap.read()
#     img=np.asarray(imagOriginal)
#     img=cv2.resize(img,(32,32))
#     img=preProcessing(img)
#     cv2.imshow("proceessed Image",img)
#     img=img.reshape(1,32,32,1)
#     #predicting.....
#     classIndex=int(model.predict_classes(img))
#     print(classIndex)
#     prediction=model.predict(img)
#     probability_value=np.argmax(prediction,axis=-1)
#     print(classIndex,probability_value)
#
#     if probability_value>0.65:
#       cv2.putText(imagOriginal,str(classIndex) + "   "+str(probability_value),
#                         (50,50),cv2.FONT_HERSHEY_COMPLEX,
#                         1,(0,0,255),1)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
