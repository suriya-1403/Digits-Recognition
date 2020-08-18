import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

################################
path='data'
images=[]
test_ratio=0.2
class_number=[]
###############

list = os.listdir(path)
print(len(list))
no_of_class=len(list)

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
x_train,x_test,y_train,y_test = train_test_split(images,class_number,test_size=0.2)

print(x_train.shape)
print(x_test.shape)
