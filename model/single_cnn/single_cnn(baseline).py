# Load Packages
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_addons as tfa
import cv2


# 1. Load Data And Data Reprocessing
### 1.1. Resource Paths
img_path = './data/img'
train_df_path = './data/train_df.csv'
test_df_path = './data/test_df.csv'
valid_df_path = './data/valid_df.csv'

### 1.2. load Dataframe Information
train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)
valid_df = pd.read_csv(valid_df_path)
print(train_df)

### 1.3. Set Image Size(width, height)
image_width = 128
image_height = 128

### 1.4. Load Images
def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (image_width,image_height))
    return img

train_png = [ img_path + x for x in train_df['file_name'] ]
test_png = [ img_path + x for x in test_df['file_name'] ]
valid_png = [img_path + x for x in valid_df['file_name'] ]
print(train_png[0])

### 1.5. Check Image Loaded
plt.imshow(train_imgs[0])

### 1.6. Image Rescale
train_imgs = [ x/255.0 for x in train_imgs ]
test_imgs = [ x/255.0 for x in test_imgs ]
valid_imgs = [ x/255.0 for x in valid_imgs ]

### 1.7. Create Labels For Training and Testing Model
### with one-hot encoding
train_labels = train_df["label"]
test_labels = test_df['label']
valid_labels = valid_df['label']

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]
test_labels = [label_unique[k] for k in test_labels]
valid_labels = [label_unique[k] for k in valid_labels]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
valid_labels = np.array(valid_labels)

train_labels = tf.keras.utils.to_categorical(train_labels, 30)
test_labels = tf.keras.utils.to_categorical(test_labels, 30)
valid_labels = tf.keras.utils.to_categorical(valid_labels, 30)
print(len(train_labels))
print(len(label_unique))
print(valid_labels)


train_imgs = np.array(train_imgs)
test_imgs = np.array(test_imgs)
valid_imgs = np.array(valid_imgs)




# 2. CNN(Convolutional Neural Networks) Model Architect
### using 5 Convolutional Layers
### using max pooling layers
### kernel size : 3,3
### activation function : relu
### activation function(output layers) : softmax

### 2.1. Create Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(image_width,image_height, 3), activation='relu'))
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
model.add(Dense(30, activation='softmax'))

### show model details
print(model.summary())

### 2.2. Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tfa.metrics.F1Score(num_classes=30, average='macro')])

### 2.3. Create Callbacks
### Create CheckPointer for Saving Best Model
modelpath = "./model/{epoch:02d}-{val_loss:.4f}-128-(5).hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

### Create EarlyStopping Callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

### 2.4. Model Train
history = model.fit(train_imgs, train_labels, validation_data=(valid_imgs, valid_labels), epochs=100,callbacks=[early_stopping_callback,checkpointer])




# 3. Model Evaluation and Visualization
### 3.1. Print Model's Test Accuracy
print("\\n Test Accuracy: "+str(model.evaluate(test_imgs,test_labels)[1]))
model.save('./model/128-(5)')


### 3.2. Show Loss(Train, Test) Graph
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Test_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Train_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


### 3.3. Show Accuracy(Train, Test) Graph
y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

plt.plot(x_len, y_vacc, marker='.', c="red", label='Test_acc')
plt.plot(x_len, y_acc, marker='.', c="blue", label='Train_acc')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()


### 3.4. Show F1-Score(Train, Test) Graph
y_vaf = history.history['val_f1_score']
y_f = history.history['f1_score']

plt.plot(x_len, y_vaf, marker='.', c="red", label='Testset_acc')
plt.plot(x_len, y_f, marker='.', c="blue", label='Trainset_acc')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
