import os

from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import plot_model

import load


def pre_img(path, rows=224, cols=224, classes=2):
    x, y = load.load_data(path, rows, cols, classes)
    x = preprocess_input(x)
    return x, y


def my_model(classes=2):
    mymodel = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
    x = mymodel.output
    x = GlobalAveragePooling2D()(x)
    pre = Dense(classes, activation='softmax')(x)
    mymodel = Model(inputs=mymodel.input, outputs=pre)
    # 冻结这些层就无法训练
    # 迁移学习，用训练好的权重，重写全连接层再进行训练
    # for layer in mymodel.layers:
    #     layer.trainable = False
    return mymodel


batch_size = 16
num_epoch = 100
train_path = "train"
vali_path = "vali"
test_path = "test"


model = my_model(2)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
ckpt = './ckpt/mobile.h5'
if os.path.exists(ckpt):
    model.load_weights(ckpt)
else:
    plot_model(model, to_file='mobile.png')
checkpoint = ModelCheckpoint(filepath=ckpt)
tensorboard = TensorBoard(log_dir='./logs/mobile', write_images=True)
train, train_label = pre_img(train_path)
vali, vali_label = pre_img(vali_path)
model.fit(x=train, y=train_label, batch_size=batch_size, epochs=num_epoch, verbose=1,
          shuffle=True, validation_data=(vali, vali_label), callbacks=[tensorboard, checkpoint])
test, test_label = pre_img(test_path)
y_pred = model.predict(test, batch_size=batch_size, verbose=1)
print(np.argmax(y_pred, axis=1))
print(test_label.argmax(axis=1))

Test_accuracy = accuracy_score(test_label.argmax(axis=-1), y_pred.argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)
