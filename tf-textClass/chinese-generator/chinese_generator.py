import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import nltk
import os
import sys
import jieba
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

'''
    文本预处理
'''

input_path = "chinese.txt"
output_path = "chinese_process.txt"

pre_text = open(input_path,'r').readlines()
after_text = open(output_path,'w')


for line in pre_text:
    after_text.write(line.strip())

# 文本分词
text = open(input_path,'r').readlines()
train_data = []
vocab_text = []
for line in text:
    train_data.append([c for c in jieba.__lcut(line)])
    for c in jieba.__lcut(line):
         vocab_text.append(c)

# 词典构建
vocab = sorted(set(vocab_text))
print(len(vocab))
print(vocab)

char2idx = {char:idx for idx,char in enumerate(vocab)}
idx2char = np.array(vocab)

def split_input_target(id_text):
    '''
        abcde -> abcd , bcde
    '''
    return id_text[0:-1],id_text[1:]

vocab_text_as_int = np.array([char2idx[c] for c in vocab_text])
print(vocab_text[0:10])
print(vocab_text_as_int[0:10])
# 构建字符数据集
char_dataset = tf.data.Dataset.from_tensor_slices(vocab_text_as_int)

# 构建句子数据集
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1,drop_remainder=True)
for ch_id in char_dataset.take(2):
    print(ch_id,idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(' '.join(idx2char[seq_id.numpy()])))

# 对句子进行划分，生成源语句和目标语句
seq_dataset = seq_dataset.map(split_input_target)
for item_input,item_output in seq_dataset.take(2):
    print(item_input.numpy())
    print(repr(' '.join(idx2char[s] for s in item_input.numpy())))
    print(repr(' '.join(idx2char[s] for s in item_output.numpy())))

# 模型构建
batch_size = 64
buffer_size = 10000

seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)
print("------------------------------------------------------------")
for item_input,item_output in seq_dataset.take(1):
    print(item_input.numpy().shape)
print("------------------------------------------------------------")
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size,embedding_dim,rnn_units,batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,
                               batch_input_shape=[batch_size,None]),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(rnn_units,stateful=True,recurrent_initializer='glorot_uniform',return_sequences = True),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(vocab_size,embedding_dim,rnn_units,batch_size)
model.summary()

def loss(labels,logits):
    return keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)

model.compile(optimizer='adam',loss = loss)

logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]

epochs = 10
model.fit(seq_dataset,epochs = epochs,steps_per_epoch=5,callbacks=callbacks)

def generator_text(model,start_string,num_generator=1000):
    input_eval = [char2idx[ch] for ch in start_string]
    input_eval = tf.expand_dims(input_eval,0)
    text_generated = []
    model.reset_states()
    for _ in range(num_generator):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predicted_ids = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
        text_generated.append(idx2char[predicted_ids])
        input_eval = tf.expand_dims([predicted_ids],0)
    return start_string+' '.join(text_generated)

new_text = generator_text(model,"哥哥")
print(new_text)
