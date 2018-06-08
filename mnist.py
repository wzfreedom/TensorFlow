# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:47:53 2018

@author: Administrator
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print ('输入数据:',mnist.train.images)
print ('输入数据打印shape:',mnist.train.images.shape)

import pylab 
im = mnist.train.images[3]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()


print ('输入数据打印shape:',mnist.test.images.shape)
print ('输入数据打印shape:',mnist.validation.images.shape)
