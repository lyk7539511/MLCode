import numpy as np  
import pandas as pd  
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from keras.datasets import mnist    
np.random.seed(10)  

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  
print("\t[Info] train data=",len(X_train_image))  
print("\t[Info] test  data=",len(X_test_image))  

print("\t[Info] Shape of train data=%s" % (str(X_train_image.shape)))  
print("\t[Info] Shape of train label=%s" % (str(y_train_label.shape)))  

import matplotlib.pyplot as plt  
def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary') # cmap='binary' 參數設定以黑白灰階顯示.  
    plt.show()  

plot_image(X_train_image[0])

y_train_label[0]

#建立 plot_images_labels_predict 函數, 可以顯示多筆資料的影像與 label. 
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14) 
    if num > 25: 
        num = 25  
    for i in range(0, num):  
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

#查看test dataset前十筆圖片：因為目前我們還沒有預測值，所以prediction傳入空陣列[]
plot_images_labels_predict(X_train_image, y_train_label, [], 0, 10)

#首先將 image 以 reshape 轉換為二維 ndarray 並進行 normalization (Feature scaling)
x_Train = X_train_image.reshape(60000, 28*28).astype('float32')  
x_Test = X_test_image.reshape(10000, 28*28).astype('float32')  
print("\t[Info] xTrain: %s" % (str(x_Train.shape)))  
print("\t[Info] xTest: %s" % (str(x_Test.shape)))  
  
#每個像素是0-255的值，影像資料標準化：針對灰階image最簡單的方式是除以255。最後print第0筆資料，發現其值已介於0與1之間。
x_Train_norm = x_Train/255  
x_Test_norm = x_Test/255  

from keras.models import Sequential  
from keras.layers import Dense,Dropout
from keras import optimizers
import keras
  
model = Sequential()  # Build Linear Model  
#normal 使用常態分佈的亂數來初始化weight權重及bias偏差。
"""
0.9481
model.add(Dense(units=1024, input_dim=784,kernel_initializer='normal', activation='relu')) # Add Input/hidden layer
model.add(Dense(units=512, input_dim=784,kernel_initializer='normal', activation='relu'))
model.add(Dense(units=1024, input_dim=784,kernel_initializer='normal', activation='relu'))
model.add(Dense(units=256, input_dim=784,kernel_initializer='normal', activation='relu'))
"""

#0.9994
model.add(Dense(784, kernel_initializer='normal', input_shape=(784,), activation='selu'))
model.add(Dense(3136, kernel_initializer='normal', activation='selu'))
model.add(Dense(3136, kernel_initializer='normal', activation='selu'))
model.add(Dense(784, kernel_initializer='normal', activation='selu'))
model.compile(loss='mean_squared_error', optimizer='adam')
#keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=0.2, nesterov=True)

#model.add(Dropout(0.5))#rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  
print("\t[Info] Model summary:")  
model.summary()  
print("")  
#Param是超參數Hyper-Parameters，指由這層神經元所產生的參數

y_TrainOneHot = np_utils.to_categorical(y_train_label)#將 training 的 label 進行 one-hot encoding
y_TestOneHot = np_utils.to_categorical(y_test_label) #將測試的 labels 進行 one-hot encoding
y_train_label[0] #檢視 training labels 第一個 label 的值

y_TrainOneHot[:1]#檢視第一個 label 在 one-hot encoding 後的結果, 會在第六個位置上為 1, 其他位置上為 0

from keras import optimizers
#在訓練模型之前, 先使用 compile 方法, 對訓練模型進行設定
#loss: 設定 loss function, 使用 cross_entropy (Cross entropy) 交叉摘順練效果較好.
#optimizer: 設定訓練時的優化方法
#metrics: 設定評估模型的方式是 accuracy (準確率)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#lr：float> = 0.學習率
#動量：浮動> = 0.參數，用於加速SGD在相關方向上前進，並抑制震盪
#decay：float> = 0.每次參數更新後學習率衰減值。
#nesterov：布爾值。是否使用涅斯捷羅夫動量。
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  

train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)
#x=x_Train_norm: features 數字的影像特徵值 (60,000 x 784 的陣列).
#y=y_Train_OneHot: label 數字的 One-hot encoding 陣列 (60,000 x 10 的陣列)
#validation_split = 0.2: 設定訓練資料與 cross validation 的資料比率.
#有訓練資料 60,000*0.8  = 48,000 ;  驗證資料  60,000*0.2  = 12,000.
#epochs = 10: 執行 10 次的訓練週期.
#batch_size = 200: mini_batch 200
#verbose = 2: 顯示訓練過程. 共執行 10 次 epoch, 每批 200 筆, 
#每次會有 240 round (48,000 / 200 = 240). 每一次的 epoch 會計算 accuracy 並記錄在 train_history 中.

import matplotlib.pyplot as plt  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss', 'val_loss')

scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

print("\t[Info] Making prediction to x_Test_norm")  
prediction = model.predict_classes(x_Test_norm)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[240:250]))  
  
plot_images_labels_predict(X_test_image, y_test_label, prediction, idx=240)  

print("\t[Info] Display Confusion Matrix:")  
import pandas as pd  
print("%s\n" % pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))    