#!/usr/bin/env python
# coding: utf-8

# In first, we need to import the necessary libraries while loading the Fashion Model to normalize and one-hot encode.

# In[ ]:


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np

# 加载 Fashion MNIST 数据集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)


# Set a list of hyperparameter combinations and store them in an array. The hyperparameter combinations include activation function, optimizer, learning rate, and dropout rate. Repeatedly test different combinations to get the best 3 sets of hyperparameter combinations.

# In[ ]:


# 超参数组合列表
hyperparams = [
    {'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.00005), 'dropout': 0.15},
    {'activation': 'relu', 'optimizer': Adam(learning_rate=0.00005), 'dropout': 0.15},
    {'activation': 'relu', 'optimizer': SGD(learning_rate=0.00005,momentum=0.9), 'dropout': 0.15},
    {'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.00005), 'dropout': 0.05},
    {'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.00001), 'dropout': 0.15},
    {'activation': 'tanh', 'optimizer': RMSprop(learning_rate=0.00005), 'dropout': 0.15},
    # 添加更多组合 add more h
]

# 存储结果
results = []


# Create and train the model.

# In[ ]:


# 构建和训练模型
for params in hyperparams:
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=params['activation']),
        Dropout(params['dropout']),
        Dense(64, activation=params['activation']),
        Dropout(params['dropout']),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=32, verbose=0)


# Record the final accuracy of the validation set and select the top three hyperparameter combinations.

# In[ ]:


# 记录验证集的最终准确率
val_accuracy = history.history['val_accuracy'][-1]
results.append({'params': params, 'val_accuracy': val_accuracy})

# 筛选出表现最好的 3 组超参数组合
top_3 = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)[:3]
print("Top 3 hyperparameter combinations:", top_3)

