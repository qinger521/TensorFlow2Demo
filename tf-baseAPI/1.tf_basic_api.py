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
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# 基础api-定义常量
t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(t)
print(t[:,1:])

# 常量的算子
print(t+10)
print(tf.square(t))
print(tf.transpose(t))

# tf-> numpy:取出值
print(t.numpy())
print(np.square(t))

# numpy->tf
np_t = np.array([[1.,2.,3.],[4.,5.,5.]])
print(tf.constant(np_t))

# tf操作字符串
s = tf.constant("cafe")
print(s)
print(tf.strings.length(s))
print(tf.strings.length(s,unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(s,"UTF8"))

'''
    raggedTensor ：不完整的二维矩阵，比如每一行的长度不一致
'''
r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
print("ragged R:",r)
print(r[1])
print(r[1:2])
print(r[:,1:2])

# ops on ragged tensor
print("-----------ops on ragged tensor----------")
r2 = tf.ragged.constant([[51,52],[],[71]])
print(tf.concat([r,r2],axis=0))


# ragged tensor -> tensor
print("----------ragged tensor -> tensor----------")
print(r.to_tensor())

# sparse tensor稀疏矩阵 传入坐标和数值
s = tf.SparseTensor(indices=[[0,1],[1,0],[2,3]],values=[1,2,3],
                    dense_shape = [3,4])
print(s)
print(tf.sparse.to_dense(s))


'''
    变量的创建
'''
print('----------变量的创建----------')
v = tf.Variable([[1,2,3],[4,5,6]])
print(v)
print(v.value())
print(v.numpy())

# assign value 重新赋值
v.assign(2*v)
print(v)

# 给单独的元素赋值
v[0,1].assign(42)
print(v)