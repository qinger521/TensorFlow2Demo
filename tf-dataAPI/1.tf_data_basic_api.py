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

# 从内存中构建数据集
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

for item in dataset:
    print(item)

'''
    对于dataset的常用操作
'''
# 1.repeat epoch
# 2.get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# 文件dataser->具体数据集
dataset2 = dataset.interleave(
    # map_fn对数据集进行怎样的操作
    lambda v : tf.data.Dataset.from_tensor_slices(v),
    # cycle_length 数据的并行程度
    cycle_length=5,
    # block_length
    block_length=5
)

for item in dataset2:
    print(item)

# 手动构建数据集
x = np.array([[1,2],[3,4],[5,6]])
y = np.array(['cat','dog','fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x,y))
print("************************DataSet3************************")
print(dataset3)

for item_x,item_y in dataset3:
    print(item_x,item_y)


dataset4 = tf.data.Dataset.from_tensor_slices({"feature":x,"label":y})
print("************************DataSet4************************")
for item in dataset4:
    print(item)