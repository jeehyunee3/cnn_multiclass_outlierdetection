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

# load Dataframe Information
df2 = pd.read_csv('./data/test_df.csv')
test_data_path = './data/mvtec_anomaly_detection'

### set parameters
num_classes=2
image_size =128

# CNN(Convolutional Neural Networks) Model Architect
### using 5 Convolutional Layers
### using max pooling layers
### kernel size : 3,3
### activation function : relu
### activation function(output layers) : softmax

### create Model
def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128,128, 3), activation='relu'))
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

# Create image data generator
def create_test_generator(num):
    test_df = pd.DataFrame({
        'file_name': [df2['file_name'][num]]
    })

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    test_generator = test_datagen.flow_from_dataframe(test_df,
                                                      test_data_path,
                                                      x_col='file_name',
                                                      y_col=None,
                                                      batch_size=64,
                                                      class_mode=None,
                                                      target_size=(128, 128))
    return test_generator


# Load models
### objects classifier model
model_15 = create_model(15)
model_15.load_weights('./data/models/128_15class_5cv_model/75-0.0000.hdf5')

### outlier classifier models
bottle_model = create_model(2)
bottle_model.load_weights('./data/models/128_bottle_5cv_model/14-0.4624.hdf5')

cable_model = create_model(2)
cable_model.load_weights('./data/models/128_cable_5cv_model/30-0.4932.hdf5')

capsule_model = create_model(2)
capsule_model.load_weights('./data/models/128_capsule_5cv_model/93-0.5122.hdf5')

carpet_model = create_model(2)
carpet_model.load_weights('./data/models/128_carpet_5cv_model/25-0.4186.hdf5')

grid_model = create_model(2)
grid_model.load_weights('./data/models/128_grid_5cv_model/08-0.4424.hdf5')

hazelnut_model = create_model(2)
hazelnut_model.load_weights('./data/models/128_hazelnut_5cv_model/11-0.3841.hdf5')

leather_model = create_model(2)
leather_model.load_weights('./data/models/128_leather_5cv_model/32-0.2112.hdf5')

metal_nut_model = create_model(2)
metal_nut_model.load_weights('./data/models/128_metal_nut_5cv_model/37-0.5225.hdf5')

pill_model = create_model(2)
pill_model.load_weights('./data/models/128_pill_5cv_model/07-0.6274.hdf5')

screw_model = create_model(2)
screw_model.load_weights('./data/models/128_screw_5cv_model/26-0.5551.hdf5')

tile_model = create_model(2)
tile_model.load_weights('./data/models/128_tile_5cv_model/11-0.4632.hdf5')

toothbrush_model = create_model(2)
toothbrush_model.load_weights('./data/models/128_toothbrush_5cv_model/07-0.5895.hdf5')

transistor_model = create_model(2)
transistor_model.load_weights('./data/models/128_transistor_5cv_model/06-0.1667.hdf5')

wood_model = create_model(2)
wood_model.load_weights('./data/models/128_wood_5cv_model/35-0.2724.hdf5')

zipper_model = create_model(2)
zipper_model.load_weights('./data/models/128_zipper_5cv_model/70-0.2093.hdf5')

### object classifier class
models_name = [bottle_model, cable_model, capsule_model, carpet_model, grid_model, hazelnut_model, leather_model, metal_nut_model, pill_model, screw_model,
               tile_model, toothbrush_model, transistor_model,wood_model,zipper_model]

### outlier classifier class
bottle_model_class = ['bottle_bad', 'bottle_good']
cable_model_class = ['cable_bad', 'cable_good']
capsule_model_class = ['capsule_bad', 'capsule_good']
carpet_model_class = ['carpet_badt', 'carpet_good']
grid_model_class = ['grid_bad', 'grid_good']
hazelnut_model_class = ['hazelnut_bad', 'hazelnut_good']
leather_model_class = ['leather_bad','leather_good']
metal_nut_model_class = ['metal_nut_bad', 'metal_nut_good']
pill_model_class = ['pill_bad', 'pill_good']
screw_model_class = ['screw_bad','screw_good']
tile_model_class = ['tile_bad', 'tile_good']
toothbrush_model_class = ['toothbrush_bad', 'toothbrush_good']
transistor_model_class = ['transistor_bad', 'transistor_good']
wood_model_class = ['wood_bad', 'wood_good']
zipper_model_class = ['zipper_bad', 'zipper_good']

class_name = [bottle_model_class, cable_model_class,capsule_model_class,carpet_model_class,grid_model_class,hazelnut_model_class,leather_model_class,
              metal_nut_model_class,pill_model_class,screw_model_class,tile_model_class,toothbrush_model_class,transistor_model_class,wood_model_class,
              zipper_model_class]
                    
file_index = []
result = []

# Model`s connection and testing
for i in range(0, len(df2)):
    gen = create_test_generator(i)
    x = model_15.predict_generator(gen)
    y = numpy.argmax(x, axis = 1)
    pred = models_name[int(y)].predict_generator(gen)
    prediction = numpy.argmax(pred, axis = 1)
    #file_index.append(df2['file_name'][i])
    file_index.append(i)
    result.append(class_name[int(y)][int(prediction)])
    #print(y,'th ',str(models_name[int(y)]), ' and ',class_name[int(y)][int(prediction)] )

### save result to dataframe
df3 = pd.DataFrame({
        'index': file_index,
        'label' : result
    })


print(df3.info())
df3.to_csv('submission.csv',encoding='utf-8-sig',index=False)

print(len(df2))
print(len(df3))

### evaluate accuracy
sum_true = 0
sum_false = 0

for i in range(0, len(df2)):
    if df2['label'][i] == df3['label'][i]:
        sum_true = sum_true + 1
    else:
        sum_false = sum_false + 1
        
print(sum_true)
print(sum_false)

print('accuracy : ',sum_true/len(df2))