# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:22:06 2018

@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

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
training_epochs = 75
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path="log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化变量
    
    #迭代处理
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #调用优化器
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            
            avg_cost += c/total_batch#均差
            
        #打印误差信息
        if (epoch+1) % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))
            
    print("Finished!")


    #测试模型
    correct_predition = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算
    accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    #保存模型
    save_path = saver.save(sess,model_path)
    print("Model saved in file : %s" % save_path)







