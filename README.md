# loss函数的 选择
* 均方误差  回归
    > mean_squared_error
* 交叉熵   二分类
    > binary_crossentropy
 
# Keras 迁移学习
```
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.layers import Dense


#【0】InceptionV3模型，加载预训练权重,不保留顶层的三个全连接层
base_model = InceptionV3(weights='imagenet', include_top=False)

#【1】增加一个空域全局平均池化层,增加全连接层
x = GlobalAveragePooling2D()(x)
x = base_model.output

x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

#【2】合并层，构建一个待 fine-turn 的新模型
model = Model(inputs=base_model.input, outputs=predictions)

#【3】冻结特征提取层（从 InceptionV3 copy来的层）, 如果训练自己的模型则可不冻结
for layer in base_model.layers:
    layer.trainable = False

#【4】冻结层后，编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#【5】训练，预测
model.fit(train, train_label, batch_size=batch_size, epochs=num_epoch, verbose=1,
                 shuffle=True, validation_data=(vali,vali_label))
model.predict(test, batch_size=batch_size, verbose=1)
```