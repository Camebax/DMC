import cv2, os, shutil, pickle, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def clear_directory(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)


def rename_learn_data():
    i = 0
    for the_dir in os.listdir('learn_data'):
        shutil.copytree('learn_data/{}'.format(the_dir), 'renamed_learn_data/{}'.format(the_dir))
        os.rename('renamed_learn_data/{}'.format(the_dir), 'renamed_learn_data/{}'.format(i))
        i += 1


clear_directory('renamed_learn_data')
rename_learn_data()

testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,4)
path = 'renamed_learn_data'
images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)
print('Total No of Classes detected: ', noOfClasses, '\n')

for the_dir in os.listdir(path):
    for the_file in os.listdir(path+'/'+the_dir):
        curImg = cv2.imdecode(np.fromfile(path+'/'+the_dir+'/'+the_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # print(the_dir)
        # print(the_file)
        # print(' ')
        # print(curImg.shape)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(the_dir)
    print(the_dir, end=" ")
    # time.sleep(0.1)
print('\n', 'Total No of elements: ', len(classNo))

# Converting array with images into numpy array
images = np.array(images)
# Converting array with names of classes into numpy array
classNo = np.array(classNo)
print(images.shape)
# print(classNo.shape)


##### Splitting the data #####
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)


numOfSamples = [] # Число символов того или иного типа
directories = []
for the_dir in os.listdir('learn_data'):
    directories.append(the_dir) # Список для нанесения обозначений на ось X
    # print(len(np.where(y_train==the_dir)[0]))
    numOfSamples.append(len(np.where(y_train==the_dir)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(directories,numOfSamples)
plt.title("No of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Num of images")
#plt.show()


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


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)
# print(y_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                             imageDimensions[1],
                                                             1), activation='relu'
                                                                )))

    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 1
stepsPerEpochVal = len(X_train)//batchSizeVal
print('Len X_train = ',len(X_train))

history = model.fit(dataGen.flow(X_train,y_train,
                                batch_size=batchSizeVal),
                                steps_per_epoch=stepsPerEpochVal,
                                epochs=epochsVal,
                                validation_data=(X_validation,y_validation),
                                shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

# pickle_out = open("model_trained.p","wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()







