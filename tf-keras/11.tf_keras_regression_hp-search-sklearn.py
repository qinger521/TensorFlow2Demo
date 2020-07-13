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

# sklearn进行超参数随机搜索
'''
    1、转化为sklearn的model
    2、定义参数集合
    3、使用RandomizedSearchCV随机搜索参数
'''
# 参数列表为所要搜索的参数
def build_model(hidden_layers=1,layer_size=30,learning_rate=3e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer_size,activation='relu',
                                    input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers-1):
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
logdir = './11.callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]
# history = sklearn_model.fit(x_train_scaled,y_train,epochs=100,
#             validation_data = (x_valid_scaled,y_valid),
#                             callbacks=callbacks)
param_distribution = {
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate": reciprocal(1e-1,1e-2) # 学习率的值为一个分布
}
random_search_cv = RandomizedSearchCV(sklearn_model,param_distribution,
                                      n_iter=10,
                                      n_jobs=5)
random_search_cv.fit(x_train_scaled,y_train,epochs=100,
                     validation_data = (x_valid_scaled,y_valid),
                     callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()


