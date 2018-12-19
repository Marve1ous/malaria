############Load libraries#####################################################
import cv2
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import image


def load_data(path, img_rows_orig=100, img_cols_orig=100, num_classes=2):
    # Load training images
    train_data_dir = path
    labels = os.listdir(train_data_dir)
    num = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        num = num + len(image_names_train)
    print(num)

    X_train = np.ndarray((num, img_rows_orig, img_cols_orig, 3), dtype=np.float32)
    Y_train = np.zeros((num,), dtype='uint8')

    i = 0
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        for image_name in image_names_train:
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            img = np.resize(img, (img_rows_orig, img_cols_orig, 3))
            X_train[i] = img
            Y_train[i] = j
            i += 1
        j += 1
    print(i)
    print('Loading done.')

    print('Transform targets to keras compatible format.')
    Y_train = np_utils.to_categorical(Y_train[:num], num_classes)

    # np.save('imgs_train.npy', X_train, Y_train)
    return X_train, Y_train


def data(train_path, vali_path, test_path):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    vali_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = vali_datagen.flow_from_directory(
        vali_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode=None)

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50,
    #     validation_data=validation_generator,
    #     validation_steps=800)

    # model.predict_generator(
    #     test_generator
    # )
