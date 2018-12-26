import os

from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils import plot_model


def get_custom_model(classes=2):
    def preprocess_input(img):
        img = img / 255.
        return img.astype(np.float32)

    def decode_img(img):
        img = img * 255.
        return img.astype(np.uint8)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # 用全局平均池化代替flatten减少参数,避免过拟合，提高正确率
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    model.summary()

    ckpt = './ckpt/custom.h5'
    checkpoint = ModelCheckpoint(filepath=ckpt)
    tensorboard = './log/custom'
    tensorboard = TensorBoard(log_dir=tensorboard)
    if os.path.exists(ckpt):
        model.load_weights(ckpt, by_name=True)
        print("load done")
    else:
        plot_model(model, to_file='custom.png')

    sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9,
              nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model, checkpoint, tensorboard, preprocess_input, decode_img
