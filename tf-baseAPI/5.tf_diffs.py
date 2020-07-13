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

'''
    自定义求导
'''
def f(x):
    return 3. * x ** 2 + 2. * x - 1

# 近似求导
def approximae_derivative(f,x,eps = 1e-6):
    return (f(x+eps)-f(x-eps)) / (2 * eps)

print(approximae_derivative(f,1.))

# 求偏导
def g(x1,x2):
    return (x1 + 5) * (x2 ** 2)

def approximae_gradient(g,x1,x2,eps=1e-6):
    dg_x1 = approximae_derivative(lambda x:g(x,x2),x1,eps)
    dg_x2 = approximae_derivative(lambda x:g(x1,x),x2,eps)
    return dg_x1,dg_x2

print(approximae_gradient(g,2.,3.))



'''
    tf.GradientTape使用方法,可自动求导，但生成的tape对象只能使用一次
'''
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent = True) as tape:
    z = g(x1,x2)

dz_x1 = tape.gradient(z,x1)
print(dz_x1)
dz_x2 = tape.gradient(z,x2)
print(dz_x2)
del tape

'''
    求解二阶导数
'''

x1 = tf.Variable(2.)
x2 = tf.Variable(3.)

with tf.GradientTape(persistent = True) as outer_tape:
    with tf.GradientTape(persistent = True) as inner_tape:
        z = g(x1,x2)
    inner_grads = inner_tape.gradient(z,[x1,x2])
outer_grads = [outer_tape.gradient(inner_grad,[x1,x2]) for inner_grad in inner_grads]
print(outer_grads)
del inner_tape
del outer_tape


'''
    手动实现梯度下降
'''
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z,x)
    x.assign_sub(learning_rate * dz_dx)
print(x)



