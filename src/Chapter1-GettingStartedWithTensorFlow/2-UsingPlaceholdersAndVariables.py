# coding:utf-8
'''
Created on 2017/10/16 下午12:02

@author: liucaiquan

Page 7:(26/370)
'''
import tensorflow as tf
import numpy as np

my_var = tf.Variable(tf.zeros([2, 3]))
sess = tf.Session()
# we have to tell TensorFlow when to initialize the variables we have created(P28)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print my_var

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2, 2])
y = tf.identity(x)
x_vals = np.random.rand(2, 2)
sess.run(y, feed_dict={x: x_vals})
print y
