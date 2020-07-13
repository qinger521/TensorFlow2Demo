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

train_file = "./csv/train.csv"
eval_file = "./csv/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head())
print(eval_df.head())

y_train = train_df.pop("survived")
y_eval = eval_df.pop("survived")

print(train_df.shape)
print(eval_df.shape)

# sex的分布直方图
train_df.sex.value_counts().plot(kind = 'barh')

# 男性女性分别有多少比例获救
print(pd.concat([train_df,y_train],axis=1).groupby("sex").survived.mean())

'''
    feature_column的使用,其将特征分为两类，离散和连续，离散特征可以方便的进行one-hot编码
'''
# 离散特征
categorical_columns = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
# 连续特征
numeric_columns = ['age','fare']
feature_columns = []

for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    feature_columns.append(
        tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(categorical_column,vocab))) # 生成one-hot编码

for numeric_column in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(numeric_column,dtype=tf.float32))


def make_dataset(data_df,label_df,epochs=10,shuffle=True,batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset
train_dataset = make_dataset(train_df,y_train,batch_size=5)
# 保存中间输出的模型
output_dir = 'baseline_model'
for x,y in train_dataset.take(1):
    print("--------------------------------------------------")
    print(x)
    print("--------------------------------------------------")
    print(y)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

baseline_estimator = tf.estimator.BaselineClassifier(model_dir=output_dir,
                                                      n_classes=2)
#baseline_estimator.train(input_fn= lambda : make_dataset(train_df,y_train,epochs=100))
baseline_estimator.train(input_fn=lambda : make_dataset(train_df,y_train,epochs=100))
baseline_estimator.evaluate(input_fn=lambda : make_dataset(eval_df,y_eval,epochs=1,shuffle=False))