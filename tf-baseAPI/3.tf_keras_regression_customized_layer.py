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

'''
    customized dense layer
'''
class CustomizedDenseLayer(tf.keras.layers.Layer):
    def __init__(self,units,activation=None,**kwargs):
        self.units = units
        self.activation = tf.keras.layers.Activation(activation)
        super(CustomizedDenseLayer,self).__init__(**kwargs)

    def build(self,input_shape):
        '''
            构建所需要的参数,
            x * w + b
            input_shape : [none, a]  output_shape : [none, units] 所以w的shape为[a,units]
            initializer : 定义参数的初始化方法
        '''
        self.kernel = self.add_weight(name='kernel',shape = ([input_shape[1],self.units]),
                                      initializer='uniform',
                                      trainable = True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True
                                    )
        super(CustomizedDenseLayer,self).build(input_shape)

    def call(self,x):
        '''
            完成正向计算   @表示转置相乘
        '''
        return self.activation(x @ self.kernel + self.bias)

# 通过lambda方式定义层次->激活函数为例
customized_softplus = tf.keras.layers.Lambda(lambda x : tf.nn.softplus(x))


# 使用子类API
class WideDeepModel(tf.keras.models.Model):
    # 定义模型层次
    def __init__(self):
        super(WideDeepModel, self).__init__()
        self.hidden1_layer = CustomizedDenseLayer(30,activation='relu',input_shape = x_train.shape[1:])
        self.hidden2_layer = CustomizedDenseLayer(30,activation='relu')
        self.dropout = tf.keras.layers.AlphaDropout(rate = 0.5)
        self.output_layer = CustomizedDenseLayer(1)


    # 完成正向传播
    def call(self,input):
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = tf.keras.layers.concatenate([input,hidden2])
        output = self.output_layer(concat)
        output = customized_softplus(output)
        return output

model = WideDeepModel()
# model.build(input_shape = x_train.shape[1:])

def customized_mse(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))

model.compile(loss=customized_mse,
              optimizer = "adam",
              metrics = ["accuracy","mse"])


history = model.fit(x_train_scaled,y_train,epochs=200,
            validation_data=(x_valid_scaled,y_valid))

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()

plot_learning_curves(history)