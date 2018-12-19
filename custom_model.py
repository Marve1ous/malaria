###############################################################################
# this code helps to train a sequential custom model on the dataset of your interest and
# visualize the confusion matrix, ROC and AUC curves
###############################################################################
# load the libraries
import os
import time

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

from load_data import load_resized_training_data, load_resized_validation_data

#########################image characteristics#################################
img_rows = 100  # dimensions of image
img_cols = 100
channel = 3  # RGB
num_classes = 2
batch_size = 32
num_epoch = 600
############################################################################################################################
"""configuring the customized model"""
# 简单网络
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

ckpt = '../ckpt/custom.h5'
if os.path.exists(ckpt):
    model.load_weights(ckpt, by_name=True)
    print("load done")

# fix the optimizer
# 随机梯度下降
sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)  # try varying this for your task and see the best fit

# compile the model
model.compile(optimizer=sgd,
              # loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # 正确率评估
print(model.summary())
exit(0)
###############################################################################
# load data
X_train, Y_train = load_resized_training_data(img_rows, img_cols)
# print the shape of the data
print(X_train.shape, Y_train.shape)
###############################################################################
# train the model
t = time.time()
# 训练
# iterator = all / batch_size
checkpoint = ModelCheckpoint(filepath=ckpt, save_weights_only=True)
tensorboard = TensorBoard(log_dir='./logs/custom')
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                 shuffle=True, validation_data=None, callbacks=[checkpoint, tensorboard])
# compute the training time
print('Training time: %s' % (time.time() - t))
###############################################################################
# predict on the validation data
X_test, Y_test = load_resized_validation_data(img_rows, img_cols)
print(X_test.shape, Y_test.shape)

# Make predictions
print('-' * 30)
print('Predicting on the validation data...')
print('-' * 30)
# 预测结果
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
print(np.argmax(y_pred, axis=1))
print(Y_test.argmax(axis=1))
# compute the accuracy
# 计算正确率
Test_accuracy = accuracy_score(Y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)
