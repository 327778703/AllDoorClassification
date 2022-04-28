# -*- coding: utf-8 -*-
# 双线性模型Bilinear Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Reshape, Lambda
from tensorflow.keras import backend as K
from CustomModel import CustomModel


def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)

def l2_norm(x):
    return K.l2_normalize(x, axis=-1)

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])

class MyBilinearModel():
    def __init__(self, inputs):
        self.inputs = inputs

    def CreateMyModel(self):
        # data_augmentation = keras.Sequential([
        #     keras.layers.experimental.preprocessing.RandomFlip()
        # ])
        # x = data_augmentation(self.inputs)
        model_vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=self.inputs)
        model_vgg16.summary()
        cnn_out_a = model_vgg16.layers[-2].output
        cnn_out_shape = model_vgg16.layers[-2].output_shape
        print(cnn_out_shape.shape)
        cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2],
                             cnn_out_shape[-1]])(cnn_out_a)
        cnn_out_b = cnn_out_a
        cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
        cnn_out_dot = Reshape([cnn_out_shape[-1] * cnn_out_shape[-1]])(cnn_out_dot)

        sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
        l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
        output = Dense(64, activation='softmax', name='my_output')(l2_norm_out)

        model = CustomModel(self.inputs, output)
        return model


