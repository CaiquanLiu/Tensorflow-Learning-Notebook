# coding:utf-8
'''
Created on 2017/10/16 下午2:02

@author: liucaiquan
Page 16:(35/370)
'''
import tensorflow as tf

sess = tf.Session()

print(sess.run(tf.nn.relu([-3., 3., 10.])))
