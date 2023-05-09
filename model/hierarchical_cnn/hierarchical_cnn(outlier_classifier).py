# -*- coding: utf-8 -*-
# Load Packages
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Load Data And Data Reprocessing
### load Dataframe Information
train_df = pd.read_csv('./data/train_df.csv')
valid_df = pd.read_csv('./data/vaild_df.csv')
test_df = pd.read_csv('./data/test_df.csv')

### set parameters
num_classes=2
image_size =128
train_data_path = './data/mvtec_anomaly_detection'
valid_data_path = './data/mvtec_anomaly_detection'

# CNN(Convolutional Neural Networks) Model Architect
### using 5 Convolutional Layers
### using max pooling layers
### kernel size : 3,3
### activation function : relu
### activation function(output layers) : softmax

### create model
def clfModel(image_size, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(image_size,image_size, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    return model

# Create train dataset and validation dataset
### Class modification is needed during training
train_file_name = []
train_state = []
vali_file_name = []
vali_state = []
for i in range(0, len(train_df)):
    if train_df['class'][i] == 'bottle' :
        train_file_name.append(train_df['file_name'][i])
        train_state.append(train_df['state'][i])
    
for i in range(0, len(valid_df)):
    if valid_df['class'][i] == 'bottle':
        vali_file_name.append(valid_df['file_name'][i])
        vali_state.append(valid_df['state'][i])

train_df = pd.DataFrame({
    'file_name': train_file_name,
    'state': train_state
})
vali_df = pd.DataFrame({
    'file_name' : vali_file_name,
    'state': vali_state
})


# Create image data generator
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
vali_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_data_path,
                                                    x_col='file_name',
                                                    y_col='state',
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  target_size=(image_size, image_size))
validation_generator =  vali_datagen.flow_from_dataframe(vali_df,
                                                         valid_data_path,
                                                         x_col='file_name',
                                                         y_col='state',
                                                       batch_size=64,
                                                       class_mode='categorical',
                                                       target_size = (image_size, image_size))

# Check GPU
tf.config.list_physical_devices('GPU')

# Model compilation and running
model = clfModel(image_size, num_classes)
with tf.device('/device:GPU:0'):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

    modelpath = "./data/models/"+str(image_size)+"_bottle_5cv_model/{epoch:02d}-{val_loss:.4f}.hdf5"

    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=100,callbacks=[early_stopping_callback,checkpointer])

    print("\n Test Accuracy: %.4f" % (model.evaluate(validation_generator)[1]))
    y_vloss = history.history['val_loss']

    y_loss = history.history['loss']

# Drawing graph of loss, accuracy and f1-score
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

plt.plot(x_len, y_vacc, marker='.', c="red", label='Testset_acc')
plt.plot(x_len, y_acc, marker='.', c="blue", label='Trainset_acc')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

y_vaf = history.history['val_f1_score']
y_f = history.history['f1_score']

plt.plot(x_len, y_vaf, marker='.', c="red", label='Testset_f1')
plt.plot(x_len, y_f, marker='.', c="blue", label='Trainset_f1')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('f1')
plt.show()