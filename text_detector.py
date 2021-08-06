import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

testRatio = 0.2
valRatio = 0.2
path = 'learn_data'
images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)
print('Total No of Classes detected: ', noOfClasses, '\n')

for the_dir in os.listdir('learn_data'):
    for the_file in os.listdir('learn_data/'+the_dir):
        curImg = cv2.imdecode(np.fromfile('learn_data/'+the_dir+'/'+the_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # print(the_dir)
        # print(the_file)
        # print(' ')
        # print(curImg.shape)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(the_dir)
    print(the_dir, end=" ")
print(len(classNo))

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)


##### Splitting the data #####
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
directories = []
for the_dir in os.listdir('learn_data'):
    directories.append(the_dir)
    # print(len(np.where(y_train==the_dir)[0]))
    numOfSamples.append(len(np.where(y_train==the_dir)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(directories,numOfSamples)
plt.title("No of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Num of images")
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[10])
# img = cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])
print(X_train.shape)
