# coding:utf-8
'''
Created on 2017/10/16 下午2:38

@author: liucaiquan
'''
import numpy as np

my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])

x_vals = np.array([my_array, my_array + 1])

print x_vals
