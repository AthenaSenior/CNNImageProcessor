
####### EGEMEN ÖNER - 18070006024 ######
####### SE 3368 - SOFT COMPUTING
####### FINAL PROJECT ###########
####### SOURCE CODE - ImagePredictor.py ######

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
import os
from keras.models import load_model
 
path = 'BaseData/validation/Accepted'
rpath = 'BaseData/validation/Rejected'

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=32,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

cnn.load_weights('model.h5')

count = 0
accepted=0
rejected=0
accept_count=0
reject_count=0

############ TESTING ###############
print("Accepted Folder Test")
print("Prediction"+ "                " + "Value must be")
for img in os.listdir(path):
    img = os.path.join(path, img)
    img = image.load_img(img, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    val = cnn.predict(img)
    print(str(np.floor(val[0][0]))+"                "+str(0))
    count = count + 1
    accept_count = accept_count+1
    if (np.floor(val[0][0]) == 0):
        accepted = accepted + 1


print("--------------------------------------")
print("Rejected Folder Test")
print("Prediction"+ "                " + "Value must be")
for img in os.listdir(rpath):
    img = os.path.join(rpath, img)
    img = image.load_img(img, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    val = cnn.predict(img)
    print(str(np.floor(val))+"                "+str(1))
    count = count + 1
    reject_count = reject_count +1
    if (np.floor(val) == 1 ):
        rejected = rejected + 1

print("Success Rate: " + str(((accepted+rejected)/count)*100)+ " %")
print("Success Rate of Only Accept: " + str(((accepted*100)/accept_count))+ " %")
print("Success Rate of Only Reject: " + str(((rejected*100)/reject_count))+ " %")
