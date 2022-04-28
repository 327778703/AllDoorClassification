# -*- coding: utf-8 -*-
# 平均值和标准差已经通过计算得到结果

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class MeanStd():
    def Getmean_std(self):
        train_mean = tf.constant([148.6807, 134.86488, 121.56722], shape=(3,), dtype=tf.float32)
        train_std = tf.constant([58.068584, 62.13803, 66.39151], shape=(3,), dtype=tf.float32)
        return train_mean, train_std