import os

import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import plot_model


def get_mobilev2_model(classes=2):
    def preprocess_input(image):
        image /= 128.
        image -= 1.
        return image.astype(np.float32)

    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    pre = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=pre)
    model.summary()
    # 冻结这些层就无法训练
    # 迁移学习，用训练好的权重，重写全连接层再进行训练
    for layer in base_model.layers:
        layer.trainable = False

    ckpt = './ckpt/mobilev2.h5'
    checkpoint = ModelCheckpoint(filepath=ckpt)
    tensorboard = './log/mobilev2'
    tensorboard = TensorBoard(log_dir=tensorboard)
    if os.path.exists(ckpt):
        model.load_weights(ckpt)
        print('load done')
    else:
        plot_model(model, to_file='mobilev2.png')

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9,
              nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model, checkpoint, tensorboard, preprocess_input
