# coding:utf-8
'''
Created on 2017/10/16 下午3:04

@author: liucaiquan

Page 35:(54/370)
'''
import matplotlib.pyplot as plt
import tensorflow as tf

sess=tf.Session()

x_vals = tf.linspace(-1., 1., 500)
# print x_vals
target = tf.constant(0.)
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)
# print l2_y_out
# print type(l2_y_out)
x_array = sess.run(x_vals)
# print x_array
# print type(x_array)
# print x_array.shape
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.show()