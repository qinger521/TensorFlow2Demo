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

# for module in mpl,np,pd,sklearn,tf,keras:
#     print(module.__name__,module.__version__)

# 导入数据集
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)
# pprint(housing.data[0:5])
# pprint(housing.target[0:5])

x_train_all,x_test,y_train_all,y_test = train_test_split(
    housing.data,housing.target,random_state=7
)
x_train,x_valid,y_train,y_valid = train_test_split(
    x_train_all,y_train_all,random_state=11
)

# 数据的归一化：x = (x-u)/std u:均值 std:方差

# 使用函数式API
input = tf.keras.layers.Input(shape=x_train.shape[1:])
# 使用两层的神经网络实现deep model
hidden1 = tf.keras.layers.Dense(30,activation="relu")(input)
hidden2 = tf.keras.layers.Dense(30,activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([input,hidden2])
output = tf.keras.layers.Dense(1)(concat)

# 将模型固化
model = tf.keras.models.Model(inputs = [input],outputs = [output])

model.compile(loss="mean_squared_error",
              optimizer = "sgd",
              metrics = ["accuracy"])


history = model.fit(x_train,y_train,epochs=100,
            validation_data=(x_valid,y_valid))

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()

plot_learning_curves(history)