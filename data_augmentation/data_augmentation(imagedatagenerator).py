# -*- coding: utf-8 -*-
#Import Packages
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#Fixing Randomeed
np.random.seed(5)

#1.Image Data Augmentation
###Since there are few outlier data, only outlier data is augmented.

###1.1 Loading the data frame
train_df = pd.read_csv('./data/train_df.csv') 

###1.2 Creating a generator object
train_datagen = ImageDataGenerator(rescale=1./255, 
                                  rotation_range=359,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.7, 1.3],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

###1.3 Transforming the data
###Augmenting the data according to the specified number of iterations after loading the image
for i in range(0,len(train_df)):
    if train_df['state'][i]!='good':
        img_name=train_df['file_name'][i]

        img = load_img('./data/'+img_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0

        for batch in train_datagen.flow(x, batch_size=1, save_to_dir='./data/imgAug', save_prefix=img_name.replace('.png', ''), save_format='png'):
            i += 1
            if i > 50: 
                print('--imageDataGenerator End--')
                break

#2.Augmented Dataframe Create
###Providing images divided by folder without a data frame to create a data frame

###2.1.1 Loading the augmented data
path_dir = './data/train_add'
file_list = os.listdir(path_dir)
###2.1.2 Loading the existing data frame
train_df = pd.read_csv('./data/train_df_test.csv') 

###2.2 Adding augmented data path
for i in file_list:
    tmp=i[0:5]+'.png'
    data=train_df.loc[train_df['file_name'] == tmp]
    data['file_name']=i
    train_df=train_df.append(data, ignore_index=True)

###2.3 Saving the data frame
train_df=train_df.drop('index', axis=1)
train_df.to_csv('./data/train_data.csv',encoding='utf8')

#3.Separating training and test data

###3.1 Splitting the data into 7:3 (train:test)
train, test = train_test_split(train_df, test_size=0.3, random_state=5)

###3.2 Loading the data frame
train_df = pd.read_csv('./data/train_data.csv') 

###3.3 Creating a column and inputting it according to training/test data separation.
train_df['train/test']=np.nan

for i in train['file_name']:
    tmp=train_df.index[train_df['file_name'] == i]
    train_df['train/test'][tmp]='train'

for i in test['file_name']:
    tmp=train_df.index[train_df['file_name'] == i]
    train_df['train/test'][tmp]='test'

###3.4 Saving the data frame
train_df.to_csv('./train_data(split).csv',encoding='utf8')