# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:03:43 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

train_x = np.linspace(-1,1,100)
train_y = 2*train_x +np.random.randn(*train_x.shape) * 0.3

plt.plot(train_x,train_y,'ro',label='Original data')
plt.legend()
plt.show()

#重置图
tf.reset_default_graph()

#创建模型
#占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")
#前向解构
z = tf.multiply(X,W)+b


#反向优化
cost = tf.reduce_mean(tf.square(Y-z))#平方差
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#梯度下降

#迭代
#初始化变量
init = tf.global_variables_initializer()
training_epochs = 20 
display_step = 2 

#定义变量和函数
plotdata={"batchsize":[],"loss":[]}#批次和损失值
def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

#模型保存
saver = tf.train.Saver({'weight':W,'bias':b})
savedir = "data/"#路径

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,savedir+"linermodel.cpkt")
    print("x=0.2,z=",sess.run(z,feed_dict={X:0.2}))