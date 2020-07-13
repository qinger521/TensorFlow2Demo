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
'''
    实现随机超参数搜索
'''
# 导入数据集
housing = fetch_california_housing()
# pprint(housing.data[0:5])
# pprint(housing.target[0:5])

x_train_all,x_test,y_train_all,y_test = train_test_split(
    housing.data,housing.target,random_state=7
)
x_train,x_valid,y_train,y_valid = train_test_split(
    x_train_all,y_train_all,random_state=11
)

# 数据的归一化：x = (x-u)/std u:均值 std:方差
# 数据的归一化：x = (x-u)/std u:均值 std:方差
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train
)

x_valid_scaled = scaler.transform(
    x_valid
)

x_test_scaled = scaler.transform(
    x_test
)

# 使用函数式API
# 搜索learning_rate,从[10^-4,10^-1]
learning_rates = [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
histories = []
for lr in learning_rates:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="relu",input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(lr)
    model.compile(loss="mean_squared_error",
              optimizer = optimizer,
              metrics = ["accuracy"])
    logdir = './10.callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file,
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]
    # history = model.fit(x_train_scaled,y_train,epochs=10,
    #         validation_data=(x_valid_scaled,y_valid),
    #                     callbacks =callbacks)
    history = model.fit(x_train_scaled, y_train, epochs=10,
                        validation_data=(x_valid_scaled, y_valid),
                        callbacks=callbacks)
    histories.append(history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()
for lr,history in zip(learning_rates,histories):
    print("learning_rate: ",lr)
    plot_learning_curves(history)