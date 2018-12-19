import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D

import load_data as load

batch_size = 4
num_epoch = 10
tensorboard = TensorBoard(log_dir='./logs/densenet', write_images=True)

train_img, train_label = load.load_resized_training_data(224, 224)
test_img, test_label = load.load_resized_validation_data(224, 224)

train_img[:, :, 0] = (train_img[:, :, 0] - 103.94) * 0.017
train_img[:, :, 1] = (train_img[:, :, 1] - 116.78) * 0.017
train_img[:, :, 2] = (train_img[:, :, 2] - 123.68) * 0.017

test_img[:, :, 0] = (test_img[:, :, 0] - 103.94) * 0.017
test_img[:, :, 1] = (test_img[:, :, 1] - 116.78) * 0.017
test_img[:, :, 2] = (test_img[:, :, 2] - 123.68) * 0.017

base_model = tf.keras.applications.DenseNet121(include_top=False, classes=2)
x = base_model.output
x = GlobalAveragePooling2D()(x)
pre = Dense(2, activation='softmax', name='fc1000')(x)
model = Model(inputs=base_model.input, outputs=pre)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x=train_img, y=train_label, batch_size=batch_size, epochs=num_epoch, verbose=1,
                 shuffle=True, validation_data=None, callbacks=[tensorboard])

# Make predictions
print('-' * 30)
print('Predicting on the validation data...')
print('-' * 30)
# 预测结果
y_pred = model.predict(test_img, batch_size=batch_size, verbose=1)
print(np.argmax(y_pred, axis=1))
print(test_label.argmax(axis=1))
# compute the accuracy
# 计算正确率
Test_accuracy = accuracy_score(test_label.argmax(axis=-1), y_pred.argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)
