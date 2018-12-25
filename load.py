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

    x_image = np.ndarray((num, img_rows_orig, img_cols_orig, 3), dtype=np.float32)
    x_label = np.zeros((num,), dtype='uint8')

    i = 0
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        for image_name in image_names_train:
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            img = np.resize(img, (img_rows_orig, img_cols_orig, 3))
            x_image[i] = img
            x_label[i] = j
            i += 1
        j += 1
    print(i)
    print('Loading done.')

    print('Transform targets to keras compatible format.')
    x_label = np_utils.to_categorical(x_label[:num], num_classes)
    return x_image, x_label


def get_label(path, num_classes=2):
    train_data_dir = path
    labels = os.listdir(train_data_dir)
    num = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        num = num + len(image_names_train)
    lab = np.zeros((num,), dtype='uint8')
    i = 0
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        for image_name in image_names_train:
            lab[i] = j
            i += 1
        j += 1
    lab = np_utils.to_categorical(lab[:num], num_classes)

    return lab


def data(train_path, vali_path, test_path,
         size=224, batch_size=32, preprocess_input=None):
    """
    用法:
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50,
    #     validation_data=validation_generator,
    #     validation_steps=800)
    #
    # model.predict_generator(
    #     test_generator
    # )
    #
    :param train_path:训练路径
    :param vali_path:验证路径
    :param test_path:测试路径
    :param size:图片大小
    :param batch_size:
    :param preprocess_input:预处理方法
    :return:生成器
    """
    train_datagen = ImageDataGenerator(
        # rescale=1. / 255,
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    vali_datagen = ImageDataGenerator(
        # rescale=1. / 255
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
        # rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(size, size),
        batch_size=batch_size)

    validation_generator = vali_datagen.flow_from_directory(
        vali_path,
        target_size=(size, size),
        batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(size, size),
        batch_size=batch_size,
        class_mode=None)

    test_label = get_label(test_path)

    return train_generator, validation_generator, test_generator, test_label
