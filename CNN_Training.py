import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

################################
path='data'
images=[]
test_ratio=0.2
val_ratio=0.2
class_number=[]
no_of_samples=[]
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

x_train=np.array(list(map(preProcessing,x_train)))
# img = x_train[30]
# img=cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitKey(0)
print(x_train[30].shape)
