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

# 将数据集导出为csv
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir,data,name_prefix,header=None,n_parts=10):
    path_format = os.path.join(output_dir,"{}_{:02d}.csv")
    filenmaes = []

    for file_idx,row_indices in enumerate(np.array_split(np.arange(len(data)),n_parts)):
        part_csv = path_format.format(name_prefix,file_idx)
        filenmaes.append(part_csv)
        with open(part_csv,"wt",encoding="utf-8") as f:
            if header is not None:
                f.write(header+"\n")
            for row_index in row_indices:
                f.write(",".join([repr(col) for col in data[row_index]]))
                f.write("\n")
    return filenmaes

# 将数据按照行进行merge
train_data = np.c_[x_train_scaled,y_train]
valid_data = np.c_[x_valid_scaled,y_valid]
test_data = np.c_[x_test_scaled,y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

train_filenames = save_to_csv(output_dir,train_data,"train",header_str,n_parts=20)
valid_filenames = save_to_csv(output_dir,valid_data,"valid",header_str,n_parts=10)
test_filenames = save_to_csv(output_dir,test_data,"test",header_str,n_parts=10)

pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)

'''
    使用tf.io.decode_csv读取csv文件生成dataset
    1.filenames -> dataset
    2. read file -> dataset -> datasets -> merge
    3. parse csv
'''
 # 1
filenmae_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filenmae_dataset:
    print(filename)

# 2
n_readers = 5
data_set = filenmae_dataset.interleave(
    # 从每个文件中按行读取，并跳过标题行
    lambda filename:tf.data.TextLineDataset(filename).skip(1),
    cycle_length=n_readers
)
print("**********data set**********")
for line in data_set.take(15):
    print(line.numpy())

# 3 tf.io.decode_csv(str,recoder_defaults) recoder-defaults：表示值得类型和默认值
sample_str = '1,2,3,4,5'
record_defaults = [tf.constant(0,dtype=tf.int32)]*5
parsed_fields = tf.io.decode_csv(sample_str,record_defaults)
print(parsed_fields)

def parse_csv_line(line,n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x,y


'''
    使用tf.io.decode_csv读取csv文件生成dataset
    1.filenames -> dataset
    2. read file -> dataset -> datasets -> merge
    3. parse csv
'''
def csv_reader_dataset(filenames,n_readers=5,batch_size=32,n_parse_threads=5,shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename : tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filenames,batch_size=3)
for x_batch,y_batch in train_set.take(2):
    print("X:")
    pprint.pprint(x_batch)
    print("Y:")
    pprint.pprint(y_batch)

'''
    往下使用keras训练训练集就好
'''





