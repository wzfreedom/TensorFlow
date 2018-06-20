# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:32:55 2018

@author: Administrator
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

myimg = mpimg.imread('img.png')
plt.imshow(myimg)
#plt.axis('off')
plt.show()
print(myimg.shape)

full = np.reshape(myimg,[1,750, 554, 3])
inputfull = tf.Variable(tf.constant(1.0,shape=[1,750, 554, 3]))

filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                 [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                 [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],
                                shape=[3,3,3,1]))

op = tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME')

o = tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255,tf.uint8)#归一化，和类型转换

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t,f=sess.run([o,filter],feed_dict={inputfull:full})
    
    t = np.reshape(t,[750, 554])
    
    plt.imshow(t,cmap='Greys_r')
    plt.axis('off')
    plt.show()