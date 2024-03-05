# Importing necessary libraries:
import cv2
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# from google.colab import drive

#nueral network training
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D
from tensorflow.keras.optimizers import Adam #optimisation
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Setting up the paths for the dataset:
disease_path=pathlib.Path(r"C:\Users\user\Desktop\covid\archive\Data\train") 


# Loading file paths for each class (COVID19, NORMAL, PNEUMONIA):
A=list(disease_path.glob("COVID19/*.jpg"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
B=list(disease_path.glob("NORMAL/*.jpg"))
C=list(disease_path.glob("PNEUMONIA/*.jpg"))

# Counting the number of images in each class:
len(A),len(B),len(C)

# Creating dictionaries to map classes to their respective file paths and class indices:
disease_dict={"COVID19":A,
              "NORMAL":B,
              "PNEUMONIA":C,
             }
disease_class={"COVID19":0,
              "NORMAL":1,
              "PNEUMONIA":2
              }


# preprocessing starts here
# Preprocessing: Resizing images, normalizing pixel values, and creating X (images) and y (labels) lists:
x=[]
y=[]

for i in disease_dict:
  disease_name=i
  disease_path_list=disease_dict[disease_name]
  print("Image resizing....")
  for path in disease_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=disease_class[disease_name]
    y.append(cls)

len(x)
print("complete")
x=np.array(x)
y=np.array(y)
# preprocessing ends herw

# Splitting the data into training and testing sets:

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)

len(xtrain),len(ytrain),len(xtest),len(ytest)

xtrain.shape

"""xtrain.shape,xtest.shape"""

xtrain.shape,xtest.shape

"""xtrain.shape,xtest.shape"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# Creating a base model using MobileNetV2:
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D
from keras.layers import Dropout

from tensorflow.keras.models import Model
# construct the head of the model that will be placed on top of the
# the base model
# Building a custom head model on top of the base model:
headModel = base_model.output
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(30, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=base_model.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
# Freezing the layers of the base model:
for layer in base_model.layers:
	layer.trainable = False

from tensorflow.keras.optimizers import Adam
# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
# Compiling the model:
opt = Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")

# Training the model:
model_hist=model.fit(xtrain,ytrain,epochs=5,validation_data=(xtest,ytest),batch_size=50)
# Saving the trained model:
model.save("Model.h5")
# model.save('my_model.model')
# print("hi")
# model.save('my_model.keras')



# This code essentially sets up a convolutional neural network using the MobileNetV2 architecture for classifying COVID-19,
# NORMAL, and PNEUMONIA from chest X-ray images. The code includes data preprocessing, model creation, training, and model saving.








