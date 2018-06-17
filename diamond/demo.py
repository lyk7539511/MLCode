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
train= pd.read_csv('D:\\STUDY\\MachineLearning\\Code\\diamond\\diamond.csv')    #import database
train_X = train['carats']   #x軸
train_Y = train['price']    #y軸
learning_rate = 0.01    #學習速率/步長

training_epochs = 100  #訓練次數
display_step = 1   #打印頻率
X = tf.placeholder(tf.float32)  #Tensor flow中的佔位符，暫時儲存變量，用來傳入數據
Y = tf.placeholder(tf.float32)

#W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32) 
#b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)
W = tf.Variable(tf.random_uniform([1],0,1.0),name="weight",dtype=tf.float32)
b = tf.Variable(tf.zeros([1]),name="bias",dtype=tf.float32)


pred = W*X+b

cost = tf.reduce_mean(tf.square(pred-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init =tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            
            plt.plot(train_X, train_Y, 'ro', label="Original data")
            plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
            plt.legend()
            plt.show()
            print()
            print("epoch:%d W:%f b:%f " % (epoch,sess.run(W),sess.run(b)))