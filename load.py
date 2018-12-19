############Load libraries#####################################################
import cv2
import numpy as np
import os
from tensorflow.python.keras.utils import np_utils


def load_data(path, img_rows_orig, img_cols_orig, num_classes=2):
    # Load training images
    train_data_dir = path
    labels = os.listdir(train_data_dir)
    num = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        num = num + len(image_names_train)
    print(num)

    X_train = np.ndarray((num, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_train = np.zeros((num,), dtype='uint8')

    i = 0
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        for image_name in image_names_train:
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            img = np.resize(img, (100, 100, 3))
            X_train[i] = img
            Y_train[i] = j
            i += 1
        j += 1
    print(i)
    print('Loading done.')

    print('Transform targets to keras compatible format.')
    Y_train = np_utils.to_categorical(Y_train[:num], num_classes)

    # np.save('imgs_train.npy', X_train, Y_train)
    return X_train, Y_train, num


def load_resized_data(img_rows, img_cols, path):
    X_train, Y_train, num = load_data(path, img_rows, img_cols, 2)
    # Resize trainging images

    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:num, :, :, :]])

    return X_train, Y_train


_, y = load_resized_data(100, 100, 'train')
print(_.shape)
