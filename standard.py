# -*- coding: utf-8 -*-
# 计算数据集的均值和标准差

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib
matplotlib.rc("font", family='FangSong')
from tensorflow.keras.preprocessing import image_dataset_from_directory


# tensorflow版本
print("tf.version:", tf.__version__)

# 数据集获取
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
IMG_SIZE = (256, 256)
path = r"D:\MyFiles\ResearchSubject\Alldatasets"
train_dataset = image_dataset_from_directory(path, batch_size=TRAIN_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                             seed=12, validation_split=0.1, subset='training')
valid_dataset = image_dataset_from_directory(path, batch_size=VALID_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                             seed=12, validation_split=0.1, subset='validation')
className = train_dataset.class_names  # 这里标签可以这样得到
for i in range(len(className)):
    c = re.split("_", className[i])
    className[i] = c[1]+"_"+c[2]
print("64个类：", className)

# 提前取好数据
AUTOTUNE = tf.data.AUTOTUNE
train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
valid_batch_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

# 批次数据标准化
def standardize(image_data):
    mean, var = tf.nn.moments(image_data, axes=[0, 1, 2])
    std = tf.math.sqrt(var)
    return mean, std

mean, std = tf.zeros([3, ]), tf.zeros([3, ])
for i in range(train_batch_dataset.cardinality()):
    imgs, labels = next(iter(train_batch_dataset))
    if len(labels) == TRAIN_BATCH_SIZE:
        mean1, std1 = standardize(imgs)
        mean = mean + (mean1 - mean) / (i+1)
        std = std + (std1 - std) / (i+1)
        # print(mean1, mean)
        # print(std1, std)
    else:
        mean1, std1 = standardize(imgs)
        mean = (mean*TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1) + len(labels)*mean1) / (TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1)+ len(labels)*mean1)
        std = (std*TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1) + len(labels)*std1) / (TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1)+ len(labels)*mean1)

print('train_datset, mean:{}, std:{}'.format(mean, std))
#
# mean187ALL = tf.constant([148.65565, 134.85287, 121.565216], shape=(3,), dtype=tf.float32)
# std187ALL = tf.constant([58.0717, 62.1316, 66.37813], shape=(3,), dtype=tf.float32)
# mean188 = tf.constant([153.59418, 137.22198, 121.961914], shape=(3,), dtype=tf.float32)
# std188 = tf.constant([57.457253, 63.3994, 69.01693 ], shape=(3,), dtype=tf.float32)
# mean = (mean187ALL*64*187 + 61*mean188) / (64*187+61)
# std = (std187ALL*64*187 + 61*std188) / (64*187+61)
# print(mean, std)

