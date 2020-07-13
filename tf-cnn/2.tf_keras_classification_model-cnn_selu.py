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

# for module in mpl,np,pd,sklearn,tf,keras:
#     print(module.__name__,module.__version__)

# 导入数据集
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

def show_single_image(img_arr):
    plt.imshow(img_arr,cmap="binary")
    plt.show()



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

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    # x_train : [none , 28, 28]
    x_train.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28,1)

x_valid_scaled = scaler.transform(
    # x_train : [none , 28, 28]
    x_valid.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28,1)

x_test_scaled = scaler.transform(
    # x_train : [none , 28, 28]
    x_test.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28,1)



logdir = './cnn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]
# 定义模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='selu',input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='selu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2))

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='selu'))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='selu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2))

model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='selu'))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='selu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='selu'))

model.add(tf.keras.layers.Dense(10,activation="softmax")) # 全连接层，激活函数为softmax

# 损失函数为交叉熵 优化器为adam
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
model.summary()
history = model.fit(x_train_scaled,y_train,epochs=1,
          validation_data=(x_valid_scaled,y_valid),
                    callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()

plot_learning_curves(history)