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

source_dir = "./generate_csv/"
os.listdir(source_dir)

def get_filenames_by_prefix(source_dir,prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir,filename))
    return results

train_filenames = get_filenames_by_prefix(source_dir,"train")
valid_filenames = get_filenames_by_prefix(source_dir,"valid")
test_filenames = get_filenames_by_prefix(source_dir,"test")

pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)

def parse_csv_line(line,n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x,y


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

train_set = csv_reader_dataset(train_filenames)
valid_set = csv_reader_dataset(valid_filenames)
test_set = csv_reader_dataset(test_filenames)

print(train_set.take(1))


