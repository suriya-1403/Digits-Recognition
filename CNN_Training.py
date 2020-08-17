import cv2
import numpy as np
import os

################################
path='data'
images=[]
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
        print(path+"/Number-"+str(x)+"/"+y)
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
print(class_number.shape)
