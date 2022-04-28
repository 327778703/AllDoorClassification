# -*- coding: utf-8 -*-
# 数据集：Alldatasets，标准化，训练MyVGG16模型

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')
import numpy as np
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from CustomMetrics import TopkAccuracy, ConfusionMatrixMetric
from Callback import MyCallback

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

# 数据集标准化
from MeanStd import MeanStd
mean, std = MeanStd().Getmean_std()

def standardize(image_data):
    image_data = (image_data - mean)/std
    image_data = tf.reverse(image_data, [-1])
    # 将RGB转成BGR，符合VGG16预训练模型的输入要求（预处理要求）
    # 在VGG16中预处理要求还有一条要进行中心化，但是如果采用VGG16默认的预处理方法，则中心化是以ImageNet数据集而言的，因此不能采用VGG16
    # 默认的预处理方法
    return image_data

train_dataset = train_dataset.map(lambda x, y: (standardize(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (standardize(x), y))

# for imgs, labels in train_dataset.take(1):
#     print(labels)

# 数据集反标准化
def reverse_standardize(image_data):
    image_data = tf.reverse(image_data, [-1])
    image_data = np.clip(image_data * std + mean, 0, 255)
    return image_data

# def displayImages(dataset):
#     plt.figure(figsize=(10, 10))
#     # 整个画布（包括各子图在内）的大小是1000×1000
#     for images, labels in dataset.take(1):
#         # 取一个batch的数据
#         for i in range(9):
#             # img = tf.squeeze(imags[i], 2)
#             plt.subplot(3, 3, i + 1)
#             plt.imshow(reverse_standardize(images[i]).astype('uint8'))
#             plt.title(className[labels[i]])
#             plt.axis('off')
#     plt.show()

# displayImages(train_dataset)
# displayImages(valid_dataset)

# 提前取好数据
AUTOTUNE = tf.data.AUTOTUNE
train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
valid_batch_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

# 创建模型
inputs = keras.Input(shape=(256, 256, 3), name="my_input")
from MyVGG16 import MyVGG16
model = MyVGG16(inputs).CreateMyModel()

# 加载上次结束训练时的权重
# model.load_weights(r"D:\MyFiles\ResearchSubject\door4\doorModels\VGG16-256-standard-new\cp-038-0.270-0.888-1.000-0.889-0.981-0.741-0.958-0.729.ckpt")
# print('successfully loading weights')
model.summary()

# 模型编译
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["acc", TopkAccuracy(2),
                                                                                           ConfusionMatrixMetric(64)])
EPOCHS = 100
INITIAL_EPOCH = 0

# 回调函数
myCallback = MyCallback()
cp_callback = myCallback.CheckPointCallback(path=r'D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-256-standard-new')
tensorboard_callback = myCallback.TensorboardCallback(log_dir=r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/VGG16-256-standard-new')
learningrate_callback = myCallback.LearningRateCallback()
path = r'D:\MyFiles\ResearchSubject\door4/logs/VGG16-standard-256-new/avg'
os.makedirs(path, exist_ok=True)
csv_filename = path + '/training_log.csv'
csv_callback = keras.callbacks.CSVLogger(csv_filename, separator=',', append=True)

# 模型训练
model.fit(train_batch_dataset, epochs=EPOCHS+INITIAL_EPOCH, initial_epoch=INITIAL_EPOCH, validation_data=valid_batch_dataset,
          callbacks=[cp_callback, tensorboard_callback, csv_callback])
