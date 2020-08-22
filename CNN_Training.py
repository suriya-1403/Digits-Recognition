import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.externals
import joblib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D

################################
path='data'
images=[]
test_ratio=0.2
val_ratio=0.2
class_number=[]
no_of_samples=[]
image_Dimension=[32,32]
batchsize_value=50
epochs_value=2
steps_per_epoch_value=2000
###############

_list_ = os.listdir(path)
print(len(_list_))
no_of_class=len(_list_)

#########################
# importing the folders as class
for x in range(0,no_of_class):
    picture_list=os.listdir(path+"/Number-"+str(x))
    for y in picture_list:
        current_image=cv2.imread(path+"/Number-"+str(x)+"/"+y)
        #print(path+"/Number-"+str(x)+"/"+y)
        current_image=cv2.resize(current_image,(32,32))
        images.append(current_image)
        class_number.append(x)
    print(x,end=" ")
print(" ")
#print(len(images)) = print(len(images))
#########################
#convert into numpy array

images = np.array(images)
class_number = np.array(class_number)

print(images.shape)
#print(class_number.shape)

#########################################
#spliting the data
#20%testing and 80%training
x_train,x_test,y_train,y_test = train_test_split(images,class_number,test_size=test_ratio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=val_ratio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
# x_train contains images and y_train contains ids
# as the numbers(0-9) is 10
for x in range(0,no_of_class):
    #print(len(np.where(y_train==x)[0]))
    no_of_samples.append(len(np.where(y_train==x)[0]))
print(no_of_samples)

plt.figure(figsize=(10,5))
plt.bar(range(0,no_of_class),no_of_samples)
plt.title("Number of Images in each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(x_train[30])
# img=cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitKey(0)

# x_train=np.array(lists(map(preProcessing,x_train)))
x_train= np.array(list(map(preProcessing,x_train)))
# img = x_train[30]
# img=cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitKey(0)
# print(x_train[30].shape)
x_test=np.array(list(map(preProcessing,x_test)))
x_validation=np.array(list(map(preProcessing,x_validation)))

#print(x_train.shape)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
#print(x_train.shape)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

################################################################################################
################################Augment part####################################################
################################################################################################

data_Generator = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)

data_Generator.fit(x_train)

y_train=to_categorical(y_train,no_of_class)
y_test=to_categorical(y_test,no_of_class)
y_validation=to_categorical(y_validation,no_of_class)
################################################################################################
def _model_():
    no_of_filters = 60
    size_of_filter=(5,5)
    size_of_filter2=(3,3)
    size_of_pool=(2,2)
    no_of_node=500

    model=Sequential()
    model.add((Conv2D(no_of_filters,size_of_filter,input_shape=(32,32,1),activation='relu')))
    model.add((Conv2D(no_of_filters,size_of_filter,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_of_filters//2,size_of_filter2,activation='relu')))
    model.add((Conv2D(no_of_filters//2,size_of_filter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_node,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=_model_()
print(model.summary())

history=model.fit_generator(data_Generator.flow(x_train,y_train,batch_size=batchsize_value),
                                                steps_per_epoch=steps_per_epoch_value,
                                                epochs=epochs_value,
                                                validation_data=(x_validation,y_validation),
                                                shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])

plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()
score=model.evaluate(x_test,y_test,verbose=0)
print('Test score: ',score[0])
print('Test Accuracy: ',score[1])
print(int(model.predict_classes(images)))

################################################################################################
################################storing as pickle object########################################
################################################################################################

# with open('pickle_file/model_trained','wb') as pickle_out:
#     pickle.dump(model,pickle_out)
# pickle_out.close()

# joblib.dump(model,'model_trained')
