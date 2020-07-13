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

'''
    tfrecord是一个文件格式
     -> tf.train.Example
      -> tf.train.Features -> {"key" : tf.train.Feature}
       -> tf.train.Feature -> tf.train.ByteList/FloatList/Int64List
'''
favorite_books = [name.encode('utf-8') for name in ["machine learning","cc150"]]
favorite_books_bylist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bylist)

hours_floatlist = tf.train.FloatList(value=[15.5,9.5,7.0,8.0])
print(hours_floatlist)

age = tf.train.Int64List(value=[42])
print(age)

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(bytes_list=favorite_books_bylist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age)
    }
)
print(features)
print("************************************************")
example = tf.train.Example(features=features)
print(example)
print("************************************************")
# 序列化，方便存贮
serialized_example = example.SerializeToString()
print(serialized_example)


# 存储为tfrecord文件
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir,filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

# 读取tfrecord文件
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)

expected_features = {
    "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([],dtype=tf.int64)
}

dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features
    )
    books = tf.sparse.to_dense(example["favorite_books"],default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))