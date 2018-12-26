import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.utils import plot_model


def get_densenet121_model(classes=2):
    def preprocess_input(img):
        img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
        img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
        img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017
        return img.astype(np.float32)

    def decode_img(img):
        img[:, :, 0] = (img[:, :, 0] / 0.017) + 103.94
        img[:, :, 1] = (img[:, :, 1] / 0.017) + 116.78
        img[:, :, 2] = (img[:, :, 2] / 0.017) + 123.68
        return img.astype(np.uint8)

    base_model = tf.keras.applications.DenseNet121(include_top=False, classes=2)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    pre = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(inputs=base_model.input, outputs=pre)
    model.summary()
    for layer in base_model.layers:
        layer.trainable = False

    ckpt = './ckpt/densenet121.h5'
    checkpoint = ModelCheckpoint(filepath=ckpt)
    tensorboard = './log/densenet121'
    tensorboard = TensorBoard(log_dir=tensorboard)
    if os.path.exists(ckpt):
        model.load_weights(ckpt, by_name=True)
        print("load done")
    else:
        plot_model(model, to_file='densenet121.png')

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model, checkpoint, tensorboard, preprocess_input, decode_img
