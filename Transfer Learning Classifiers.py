import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model
import tensorflow as tf

import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time

start = time.time() #Start a timer

x_train = []
y_train = []
x_test = []
test_labels = []

def datainput(path): #Function to handle extraction of data
    images_arr = []
    labels_arr = []
    for directory_path in glob.glob(path):
        label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path,"*.jpg")): #finds all files with .jpg file type
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (45, 80))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#changes colour format to more suitable
            images_arr.append(img)
            labels_arr.append(label)
            #Appends images and labels to array
    images_arr = np.array(images_arr)
    labels_arr = np.array(labels_arr)
    return(images_arr, labels_arr)

#Calls the function
x_train, y_train = datainput("./waste_dataset/augmented_waste_dataset_splited2/train/*") 
x_test, test_labels = datainput("./waste_dataset/augmented_waste_dataset_splited2/test/*")

#Normalises the data to 0 - 1 
x_train, x_test = x_train/ 255.0, x_test / 255.0
#Converts str to int for classification
label_conv = preprocessing.LabelEncoder()
label_conv.fit(y_train)
y_train = label_conv.transform(y_train)
label_conv.fit(test_labels)
y_test = label_conv.transform(test_labels)

# Loads the Trained CNN
CNN_model = tf.keras.models.load_model('./pre_trained_model.h5')    
#Freeze all layers of the model
CNN_model.trainable = False
# Show the model architecture after freezing the layers
CNN_model = Model(inputs=CNN_model.input, outputs=CNN_model.get_layer("conv2d_14").output)

#Creates the features for the classifiers to use
x_train = CNN_model.predict(x_train)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = CNN_model.predict(x_test)
x_test = x_test.reshape(x_test.shape[0], -1)
#Creates Random forest classifier, fits and predicts. Training/ Testing datasets respectively
classifier = RandomForestClassifier(n_estimators = 50, random_state = 42)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
prediction = label_conv.inverse_transform(prediction)
#computes accuaracy 
accRF = metrics.accuracy_score(test_labels, prediction)
print("Accuracy of RF = ",accRF )
#Creates SVM classifier, fits and predicts. Training/ Testing datasets respectively
classifier = svm.SVC()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
prediction = label_conv.inverse_transform(prediction)
#computes accuaracy 
accSVM = metrics.accuracy_score(test_labels, prediction)
print("Accuracy of SVM = ", accSVM)
#Creates Naive Bayes classifier, fits and predicts. Training/ Testing datasets respectively
classifier = GaussianNB()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
prediction = label_conv.inverse_transform(prediction)
#computes accuaracy 
accGNB = metrics.accuracy_score(test_labels, prediction)
print("Accuracy of GNB = ", accGNB)
#Creates KNN classifier, fits and predicts. Training/ Testing datasets respectively
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
prediction = label_conv.inverse_transform(prediction)
#computes accuaracy 
accKNN = metrics.accuracy_score(test_labels, prediction)
print("Accuracy of KNN = ", accKNN)
#prints how long it took
print ('Time taken for run {} sec\n'.format(time.time() - start))
#plots data 
objects = ('Random Forest', 'SVM', 'Naive Bayes', 'KNN')
y_pos = np.arange(len(objects))
performance = [accRF,accSVM,accGNB,accKNN]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Machine Learning Algorithms')
#prints plot
plt.show()