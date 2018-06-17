# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 22:57:01 2018

@author: Liu
"""

from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
train= pd.read_csv('D:\\STUDY\\MachineLearning\\Code\\regression\\train.csv')    #import database

#坐標系
train_X = train['carats']   #x軸
train_Y = train['price']    #y軸

#訓練參數
learning_rate = 0.003    #學習速率/步長
training_epochs = 100  #訓練次數
display_step = 1   #打印頻率

X = tf.placeholder(tf.float32)  #Tensorflow中的佔位符，暫時儲存變量，用來傳入數據
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(np.random.randn(), name="weight_1", dtype=tf.float32)
W2 = tf.Variable(np.random.randn(), name="weight_2", dtype=tf.float32)
#b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)
#W1 = tf.Variable(tf.random_uniform([1],0,1.0),name="weight_1",dtype=tf.float32)
#W2 = tf.Variable(tf.random_uniform([1],0,1.0),name="weight_2",dtype=tf.float32)
b = tf.Variable(tf.zeros([1]),name="bias",dtype=tf.float32)

y = b + W1 * X + W2 * np.square(X) #

cost = tf.reduce_mean(tf.square(y-Y))   #均方差
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)     #梯度下降，尋找最優解

init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch) % display_step == 0:
            plt.plot(train_X, train_Y, 'ro', label="Original data")  #源數據
            plt.plot(train_X, sess.run(b) + sess.run(W1) * train_X + sess.run(W2) * np.square(train_X), label="Fitted line")    #擬合綫
            plt.legend()
            plt.show()
            print()
            print("epoch:%d W1:%f W2:%f b:%f " % (epoch,sess.run(W1),sess.run(W2),sess.run(b)))