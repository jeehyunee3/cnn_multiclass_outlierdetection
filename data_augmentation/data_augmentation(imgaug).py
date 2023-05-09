# -*- coding: utf-8 -*-
#Import Packages
import os
import cv2 
import numpy as np 
import pandas as pd
import imgaug as ia 
import imgaug.augmenters as iaa

#1.Creating imgaugClass
### reference : https://keyog.tistory.com/39
class Img_aug : 
    def __init__(self) : 
        ### Sometimes(0.5, ...) applies the given augmenter in 50% of all cases, 
        ### e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image. 
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug) 
        
        ### Define our sequence of augmentation steps that will be applied to every image 
        ### All augmenters with per_channel=0.5 will sample one value _per image_ 
        ### in 50% of all cases. In all other cases they will sample new values 
        ### _per channel_. 
        self.seq = iaa.Sequential( 
            [ 
                ### apply the following augmenters to most images 
                iaa.Fliplr(0.5), ### horizontally flip 50% of all images 
                iaa.Flipud(0.2), ### vertically flip 20% of all images 
                ### crop images by -5% to 10% of their height/width 
                self.sometimes(iaa.CropAndPad( 
                    percent=(-0.05, 0.1), 
                    pad_mode=ia.ALL, 
                    pad_cval=(0, 255) 
                )), 
                self.sometimes(iaa.Affine( 
                    scale={"x": (0.7, 1.1), "y": (0.7, 1.1)}, ### scale images to 70-110% of their size, individually per axis 
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, ### translate by -10 to +10 percent (per axis) 
                    rotate=(-45, 45), ### rotate by -45 to +45 degrees 
                    shear=(-16, 16), ### shear by -16 to +16 degrees 
                    order=[0, 1], ### use nearest neighbour or bilinear interpolation (fast) 
                    cval=(0, 255), ### if mode is constant, use a cval between 0 and 255 
                    mode=ia.ALL ### use any of scikit-image's warping modes (see 2nd image from the top for examples) 
                )), 
                ### execute 0 to 5 of the following (less important) augmenters per image 
                ### don't execute all of them, as that would often be way too strong 
                iaa.SomeOf((0, 5), 
                           [ 
                               self.sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), 
                               ### convert images into their superpixel representation 
                               iaa.OneOf([ 
                                   iaa.GaussianBlur((0, 3.0)), ### blur images with a sigma between 0 and 3.0 
                                   iaa.AverageBlur(k=(2, 7)), ### blur image using local means with kernel sizes between 2 and 7 
                                   iaa.MedianBlur(k=(3, 11)), ### blur image using local medians with kernel sizes between 2 and 7 
                               ]), 
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), ### sharpen images 
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), ### emboss images 
                               ### search either for all edges or for directed edges, 
                               ### blend the result with the original image using a blobby mask 
                               iaa.SimplexNoiseAlpha(iaa.OneOf([ 
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)), 
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)), 
                               ])), 
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
                               ### add gaussian noise to images 
                               iaa.OneOf([ 
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5), ### randomly remove up to 10% of the pixels 
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2), 
                               ]), 
                               iaa.Invert(0.05, per_channel=True), ### invert color channels
                               iaa.Add((-10, 10), per_channel=0.5), ### change brightness of images (by -10 to 10 of original value) 
                               iaa.AddToHueAndSaturation((-20, 20)), ### change hue and saturation 
                               ### either change the brightness of the whole image (self.sometimes 
                               ### per channel) or change the brightness of subareas 
                               iaa.OneOf([ 
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5), 
                                   iaa.FrequencyNoiseAlpha( 
                                       exponent=(-4, 0), 
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True), 
                                       second=iaa.ContrastNormalization((0.5, 2.0)) 
                                   ) 
                               ]), 
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), ### improve or worsen the contrast 
                               iaa.Grayscale(alpha=(0.0, 1.0)), 
                               self.sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), ### move pixels locally around (with random strengths) 
                               self.sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), ### self.sometimes move parts of the image around 
                               self.sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))) 
                           ], 
                           random_order=True 
                          ) 
            ], 
            random_order=True 
        )

#2.Image Data Augmentation

###2.1 Creating a data augmentation object
aug = Img_aug() 

###2.2 Loading the data frame
save_path = './data/imgaug/' 
data=pd.read_csv('./data/train_df.csv', index_col=0)

###2.3 Transforming the data
###Data augmentation for resolving data imbalance
for i in range(0,len(data)):
    filename=data['file_name'][i]
    if data['state'][i]=='good':
        if data['class'][i]=='toothbrush':
            augment_num = 20
        else:
            augment_num = 10
    else:
        augment_num = 30
    
    img = cv2.imread('./data/train/'+filename) 
    images_aug = aug.seq.augment_images([img for i in range(augment_num)]) 

    for num,aug_img in enumerate(images_aug) : 
        cv2.imwrite(save_path+filename.replace('.png','')+'_{}.jpg'.format(num),aug_img) 
        
    print('--'+filename+' complete--')
print('--Complete augmenting images--')

#3.Separating training and test data

###3.1.1 Loading the augmented data
path_dir = './data/imgaug'
file_list = os.listdir(path_dir)
###3.1.2 Loading the existing data frame
train_df = pd.read_csv('./data/train_df_test.csv') 

###3.2 Adding augmented data path
for i in file_list:
    tmp=i[0:5]+'.png'
    print(tmp)
    data=train_df.loc[train_df['file_name'] == tmp]
    data['file_name']=i
    #print(data)
    train_df=train_df.append(data, ignore_index=True)

###3.3 Saving the data frame
train_df=train_df.drop('index', axis=1)
train_df.to_csv('./data/train_data.csv',encoding='utf8')

#4.Separating training and test data

###4.1 Splitting the data into 7:3 (train:test)
train, test = train_test_split(train_df, test_size=0.3, random_state=5)

###4.2 Loading the data frame
train_df = pd.read_csv('./data/train_data.csv')

###4.3 Creating a column and inputting it according to training/test data separation.
train_df['train/test']=np.nan

for i in train['file_name']:
    tmp=train_df.index[train_df['file_name'] == i]
    train_df['train/test'][tmp]='train'

for i in test['file_name']:
    tmp=train_df.index[train_df['file_name'] == i]
    train_df['train/test'][tmp]='test'

###4.4 Saving the data frame
train_df.to_csv('./train_data(split).csv',encoding='utf8')

