import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import image


def load_data(path, pre_input, image_size=100, num_classes=2):
    train_data_dir = path
    labels = os.listdir(train_data_dir)
    num = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        num = num + len(image_names_train)
    print(num)

    x_image = np.ndarray((num, image_size, image_size, 3), dtype=np.float32)
    x_label = np.zeros((num,), dtype='uint8')

    i = 0
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        for image_name in image_names_train:
            img_path = os.path.join(path, label, image_name)
            img = image.load_img(path=img_path, target_size=(image_size, image_size))
            img = np.array(img, np.uint8)
            img = pre_input(img)
            x_image[i] = img
            x_label[i] = j
            i += 1
        j += 1
    print(i)
    print('Loading done.')
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


def data(train_path, vali_path, size=224,
         batch_size=32, preprocess_input=None):
    """
    用法:
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50,
    #     validation_data=validation_generator,
    #     validation_steps=800)
    # 预测返回预测值，但生成器的label不知如何获取，不建议使用
    # model.predict_generator(
    #     test_generator
    # )
    #
    :param train_path:训练路径
    :param vali_path:验证路径
    :param size:图片大小
    :param batch_size:
    :param preprocess_input:预处理方法
    :return:生成器
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    vali_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # test_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input
    #     # rescale=1. / 255
    # )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(size, size),
        batch_size=batch_size
    )

    validation_generator = vali_datagen.flow_from_directory(
        vali_path,
        target_size=(size, size),
        batch_size=batch_size
    )

    # test_generator = test_datagen.flow_from_directory(
    #     test_path,
    #     target_size=(size, size),
    #     batch_size=batch_size,
    #     shuffle=False)
    return train_generator, validation_generator

