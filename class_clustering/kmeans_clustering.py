# -*- coding: utf-8 -*-
#Import Packages
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans

#1. Data Preprocess

#1.1.1 Loading the data
img_path = './data/mvtec_anomaly_detection/' 
train_df_path = './data/mvtec_anomaly_detection/train_df.csv'
train_df = pd.read_csv(train_df_path)

#1.1.2 Loading the images
image_width =128
image_height = 128

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (image_width,image_height))
    return img

img_path = [ img_path + x for x in train_df['file_name'] ]
img_png = [ img_load(m) for m in tqdm(img_path) ]

#1.2.1 Image rescale
img_png = [ x.astype('float32')/255.0 for x in img_png ]

#1.2.2 Converting image data to numpy array
img_png = np.array(img_png)

#1.2.3 Reshaping image data
img_png_2d = img_png.reshape((len(img_png), np.prod(img_png.shape[1:])))

#2. Data clustering ; K-Means Clustering

#2.1 Creating and training the clustering algorithm
km = KMeans(n_clusters=7)
km.fit(img_png_2d)

#2.2 Printing the results by class.
train_df.loc[2]['label']
train_df['label']
np.unique(train_df['label'])
print(np.unique(km.labels_, return_counts=True))

A = dict()
B = dict()
C = dict()
D = dict()
E = dict()
F = dict()
G = dict()

result_set_list = [A, B, C, D, E, F, G]
idx = 0
for x in km.labels_ :
    label = train_df.loc[idx]['label']
    if label in result_set_list[x] :
        result_set_list[x][label] = result_set_list[x][label]+1
    else :
        result_set_list[x][label] = 1
    idx = idx + 1

print(A)
print(B)
print(C)
print(D)
print(E)
print(F)
print(G)