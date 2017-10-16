# coding:utf-8
'''
Created on 2017/10/16 下午2:08

@author: liucaiquan
Page 28(47/370)
'''
import tensorflow as tf

sess = tf.Session()

import numpy as np

x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))
