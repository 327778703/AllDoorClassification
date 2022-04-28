# -*- coding: utf-8 -*-
# VGG16，三个256全连接层，无Dropout，对应models:VGG16-256-standard

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')
from CustomModel import CustomModel


class MyVGG16():
    def __init__(self, inputs):
        self.inputs = inputs

    def CreateMyModel(self):
        # data_augmentation = keras.Sequential([
        #     keras.layers.experimental.preprocessing.RandomFlip()
        # ], name='RandomFlip')
        # preprocess_input = keras.applications.vgg16.preprocess_input
        # x = data_augmentation(self.inputs)
        # x = preprocess_input(x)

        base_model = keras.applications.VGG16(input_tensor=self.inputs, include_top=False, weights='imagenet', pooling='avg')
        # globalavg = keras.layers.GlobalAvgPool2D()
        # base_model.summary()
        base_model.trainable = False
        fc1 = keras.layers.Dense(256, activation='relu', name='dense_1')
        # drop1 = keras.layers.Dropout(0.5, name='dropout_1')
        fc2 = keras.layers.Dense(256, activation='relu', name='dense_2')
        # drop2 = keras.layers.Dropout(0.5, name='dropout_2')
        fc3 = keras.layers.Dense(256, activation='relu', name='dense_3')
        # drop2 = keras.layers.Dropout(0.5, name='dropout_2')
        fc4 = keras.layers.Dense(64, activation='softmax', name='output')
        x = base_model.output
        # x = globalavg(x)
        x = fc1(x)
        # x = drop1(x)
        x = fc2(x)
        # x = drop2(x)
        x = fc3(x)
        self.outputs = fc4(x)
        return CustomModel(self.inputs, self.outputs)
