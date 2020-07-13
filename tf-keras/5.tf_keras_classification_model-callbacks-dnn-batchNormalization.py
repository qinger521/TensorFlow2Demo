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

print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# 数据的归一化：x = (x-u)/std u:均值 std:方差
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    # x_train : [none , 28, 28]
    x_train.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28)

x_valid_scaled = scaler.transform(
    # x_train : [none , 28, 28]
    x_valid.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28)

x_test_scaled = scaler.transform(
    # x_train : [none , 28, 28]
    x_test.astype(np.float32).reshape(-1,1)
).reshape(-1,28,28)

# 神经网络实现批归一化、dropout
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.AlphaDropout(rate = 0.5)) # 相对于普通drop 此方法有两个优点：1、均值和方差不变。2、归一化性质不变
model.add(tf.keras.layers.Dense(10,activation="softmax"))


model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

# 会使用三个callback：Tensorboard、earlystopping、ModelCheckpoint
logdir = './dnn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]
history = model.fit(x_train_scaled,y_train,epochs=10,
            validation_data=(x_valid_scaled,y_valid),
            callbacks = callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()

plot_learning_curves(history)