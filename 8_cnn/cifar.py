# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:07:18 2018

@author: huangguan
"""

import cifar10_input
import tensorflow as tf 
import pylab

batch_size = 128 
data_dir = '../cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
tf.train.start_queue_runners(sess=sess)
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n",image_batch[0])

print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()
