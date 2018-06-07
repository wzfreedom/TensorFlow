# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:07:27 2018

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

#重置图---清空变量
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
tf.summary.histogram('z',z)#预测值用直方图的形式表示

#反向优化
cost = tf.reduce_mean(tf.square(Y-z))#平方差
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#梯度下降

tf.summary.scalar('loss_function',cost)#损失用标量形式显示

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
saver = tf.train.Saver(max_to_keep=1)#一个检查点
savedir = "data/"#路径

#载入检查点 2
kpt = tf.train.latest_checkpoint(savedir)

#启动SESSION
with tf.Session() as sess:
    sess.run(init)
    
    #可视化
    merged_summary_op =tf.summary.merge_all()#合并所有summary
    #创建summary_writer 写文件
    summary_writer = tf.summary.FileWriter('data/mnist_with_summmaries',sess.graph)
    
    #输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        
        #生成summary
        summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
        summary_writer.add_summary(summary_str,epoch)#写入文件
        
        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print ("Epoch:",epoch+1,"cost=",loss,"w=",sess.run(W),"b=",sess.run(b))
            
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
            #保存检查点
            #saver.save(sess,savedir+"linermodel.cpkt",global_step=epoch)
           
                
    print("finished")
    
    #保存训练好的模型--训练完保存
    saver.save(sess,savedir+"linermodel.cpkt")
    
    #print ("cost=",sess.run(cost,feed_dict={X:train_x,Y:train_y}),"w=",sess.run(W),"b=",sess.run(b))
    print ("cost:",cost.eval({X:train_x,Y:train_y}),"w=",sess.run(W),"b=",sess.run(b))
    
    #图像显示
    plt.plot(train_x,train_y,'ro',label='Original data')
    plt.plot(train_x,sess.run(W)*train_x+sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('Loss')
    plt.title('minibatch run vs training loss')
    plt.show()
    
    print("x=0.2,z=",sess.run(z,feed_dict={X:0.2}))

#载入检查点 1
#load_epoch = 18
#with tf.Session() as sess3:
#    sess3.run(tf.global_variables_initializer())
#    saver.restore(sess3,savedir+"linermodel.cpkt-"+str(load_epoch))
#    print("x=0.2,z=",sess3.run(z,feed_dict={X:0.2}))
    #载入检查点 2
#    if kpt != None:
#        saver.restore(sess,kpt)
#    print("x=0.2,z=",sess.run(z,feed_dict={X:0.2}))

#with tf.Session() as sess2:
#    sess2.run(tf.global_variables_initializer())
#    saver.restore(sess2,savedir+"linermodel.cpkt")
#    print("x=0.2,z=",sess2.run(z,feed_dict={X:0.2}))
    