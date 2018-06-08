# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:22:06 2018

@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import pylab

tf.reset_default_graph()

#定义输入变量
#占位符
x = tf.placeholder(tf.float32,[None,784])#mnist数据集 是 28*28 = 784 的图片
y = tf.placeholder(tf.float32,[None,10])#便签one_hot 0-9 数字 是个类别

#定义学习参数
w = tf.Variable(tf.random_normal([784,10]))#784*10 的矩阵
b = tf.Variable(tf.zeros([10]))

#定义运算模型
pred = tf.nn.softmax(tf.matmul(x,w)+b)#softmax分类

#优化函数，损失函数
#损失函数：计算值和实际值进行交叉熵运算，并取平均值
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

learn_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

#初始化 运行计算
training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path="log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化变量
    #读取数据模型
    saver.restore(sess,model_path)
    
    correct_prediction =  tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

    output = tf.argmax(pred,1)    
    batch_xs,batch_ys = mnist.train.next_batch(2)
    outputval,predv = sess.run([output,pred],feed_dict={x:batch_xs})
    print(outputval,predv,batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
