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

# for module in mpl,np,pd,sklearn,tf,keras:
#     print(module.__name__,module.__version__)

# 导入数据集
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test[0].shape,y_test.shape)

def show_single_image(img_arr):
    plt.imshow(img_arr,cmap="binary")
    plt.show()

show_single_image(x_train[1])

def show_imgs(n_rows,n_cols,x_data,y_data,class_names):
    assert len(x_data) == len(y_data)
    assert n_rows*n_cols < len(x_data)
    plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols *row +col
            plt.subplot(n_rows,n_cols,index+1)
            plt.imshow(x_data[index],cmap="binary",
                       interpolation="nearest")
            plt.axis("off")
            plt.title(class_names[y_data[index]])
    plt.show()

class_names = ['T-shirt','Trouser','Pullover','Dress',
               'Coat','Sandal','Shirt','Sneaker',
               'Bag','Ankle boot']

#show_imgs(3,5,x_train,y_train,class_names)
# 定义模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
model.add(tf.keras.layers.Dense(300,activation="relu")) # 全连接层 有300个节点 激活函数为relu
model.add(tf.keras.layers.Dense(100,activation="relu")) # 全连接层 100个节点 激活函数为relu
model.add(tf.keras.layers.Dense(10,activation="softmax")) # 全连接层，激活函数为softmax

# 损失函数为交叉熵 优化器为adam
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
model.summary()
history = model.fit(x_train,y_train,epochs=3,
          validation_data=(x_valid,y_valid))

data = x_test[1]
data = data[None,:]
result = model.predict(data)
print(class_names[np.argmax(result)])
print(class_names[y_test[1]])
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()

plot_learning_curves(history)