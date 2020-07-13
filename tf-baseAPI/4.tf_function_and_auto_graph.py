import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pprint

'''
    tf.function : 将python方法转化为计算图
    
'''
@tf.function
def scaled_elu(z,scale=1.0,alpha=1.0):
    '''
        z > 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    '''
    is_positive = tf.greater_equal(z,0.0)
    return scale * tf.where(is_positive,z,alpha * tf.nn.elu(z))

print(scaled_elu(tf.constant(5.)))
# scaled_elu_tf = tf.function(scaled_elu)
# print(scaled_elu_tf(tf.constant(-3.)))

# 通过scaled_elu_tf.python_function,将该方法转化为python方法

'''
    函数的签名
'''

@tf.function
def cube(z):
    return tf.pow(z,3)

print(cube(tf.constant([1.,2.,3.])))
print(cube(tf.constant([1,2,3])))

# 参数的输入可以是整型，也可为浮点型，那么如何限制输入的类型呢？ 通过函数的签名@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')]])
# tf.TensorSpec([None],tf.int32,name='x')] 含义为，首先必须是tf的变量，然后必须是int32
# 在tensorflow中必须使用函数签名才能将函数变为可保存的图结构

@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')])
def cube_int(z):
    return tf.pow(z,3)

try:
    print(cube_int(tf.constant([1.,2.,3.])))
except ValueError as erro:
    print("******************************ERROR******************************")
    print(erro)
    print("******************************ERROR******************************")

print(cube_int(tf.constant([1,2,3])))

# 与上相同，不同的实现方式
cub3_int32 = cube.get_concrete_function(tf.TensorSpec([None],tf.int32))

# 计算图
print(cub3_int32.graph)
print(cub3_int32.graph.get_operations())