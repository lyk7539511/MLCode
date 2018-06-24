# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt
import csv

train = pd.read_csv("D:\\STUDY\\MachineLearning\\Doc\\HW3\\train.csv")
train_label = train['label']
train_feature = train['feature']
print(train_label.head())

test = pd.read_csv("D:\\STUDY\\MachineLearning\\Doc\\HW3\\test.csv")
test_feature = test['feature']

train_data = np.array([np.reshape(tf.split(' '),(48,48))for tf in train_feature])
train4d = train_data.reshape(train_data.shape[0], 48, 48, 1).astype('float32')
print("\t[Info] Shape of train data=%s" % (str(train4d.shape)))
train4d_norm = train4d/255

test_data = np.array([np.reshape(tf.split(' '), (48, 48)) for tf in test_feature])
test4d = test_data.reshape(test_data.shape[0], 48, 48, 1).astype('float32')
print("\t[Info] Shape of test data=%s" % (str(test4d.shape)))
test4d_norm = test4d/255

train_oneHot = np_utils.to_categorical(train_label)
print(train_oneHot[:1])

model = Sequential()
model.add(Conv2D(filters = 144,kernel_size = (5,5),padding = 'same',input_shape = (48,48,1),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters = 144,kernel_size = (5,5),padding = 'same',input_shape = (48,48,1),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters = 144,kernel_size = (5,5),padding = 'same',input_shape = (48,48,1),activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.2)) 
model.add(Dense(7, activation='softmax'))  
model.summary()  
print("")

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
train_history = model.fit(x=train4d_norm,y=train_oneHot,validation_split=0.2,epochs=15,batch_size=300,verbose=2)

def show_trainHistory(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc = 'upper left')
    plt.show()

def plot_imageLablePredict(images,labels,prediction,idx,num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25:
        num = 25
    for i in range(0,num):
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()

show_trainHistory(train_history, 'acc', 'val_acc')
show_trainHistory(train_history, 'loss', 'val_loss')

scores = model.evaluate(train4d_norm, train_oneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

print(test4d_norm.shape)
print("\t[Info] Making prediction of test4d_norm")  
prediction = model.predict_classes(test4d_norm)  # Making prediction and save result to prediction  
print()  
print(prediction.shape)
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile,delimiter = ',')
    writer.writerow(['id','label'])
    count = 0
    for i in prediction:
        print(str(count) + "," +str(i))
        writer.writerow([count,i])
        count += 1