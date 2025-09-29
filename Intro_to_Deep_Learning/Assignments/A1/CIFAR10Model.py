#!/usr/bin/env python
# coding: utf-8

# In first, we need to import the necessary libraries while loading the CIFAR-10 Model to normalize and one-hot encode.

# In[ ]:


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)


# Based on the best hyperparameter combination obtained from the FASHION MNIST dataset, a combination test is performed to obtain the best parameters.

# In[ ]:


# 在FASHION MNIST数据集所得出最佳超参数组合基础上再进行组合测试以得出最佳参数.
#based on the best performance sets in MNIST datasets, tuning in the CIFAR-10 Model
best_hyperparams = [
    #{'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.00005), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': Adam(learning_rate=0.00005), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.00005), 'dropout': 0.15},
     #{'activation': 'relu', 'optimizer': SGD(learning_rate=0.01,momentum=0.9), 'dropout': 0.5},
     #{'activation': 'relu', 'optimizer': SGD(learning_rate=0.01,momentum=0.9), 'dropout': 0.4},
     #{'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.01), 'dropout': 0.5},
     #{'activation': 'relu', 'optimizer': Adam(learning_rate=0.01), 'dropout': 0.5},
    #{'activation': 'relu', 'optimizer': SGD(learning_rate=0.005, momentum=0.9), 'dropout': 0.3},
    #{'activation': 'relu', 'optimizer': Adam(learning_rate=0.001), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': RMSprop(learning_rate=0.001), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': SGD(learning_rate=0.005, momentum=0.9), 'dropout': 0.1},
    #{'activation': 'relu', 'optimizer': SGD(learning_rate=0.007, momentum=0.9), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': Adam(learning_rate=0.005), 'dropout': 0.2},
    #{'activation': 'relu', 'optimizer': Adam(learning_rate=0.005), 'dropout': 0.1},
    {'activation': 'relu', 'optimizer': SGD(learning_rate=0.005, momentum=0.9), 'dropout': 0.3},
    {'activation': 'relu', 'optimizer': SGD(learning_rate=0.005, momentum=0.9), 'dropout': 0.1},
    {'activation': 'relu', 'optimizer': SGD(learning_rate=0.007, momentum=0.9), 'dropout': 0.1},
]

# 存储结果
results = []


# Create and train the CNN model.

# In[ ]:


# 定义 CNN 模型结构
def create_cnn_model(input_shape, activation, dropout_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation=activation, padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(64, (3, 3), activation=activation, padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(128, (3, 3), activation=activation, padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Conv2D(256, (3, 3), activation=activation, padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Flatten(),
        Dense(512, activation=activation),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    return model



# 训练每组超参数组合
for params in best_hyperparams:
    model = create_cnn_model((32, 32, 3), params['activation'], params['dropout'])
    model.compile(optimizer=params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(x_train, y_train, epochs=25, validation_split=0.1, batch_size=32, verbose=1)


# Record the final accuracy of the validation set and select the best hyperparameter combination.

# In[ ]:


# 记录验证集的最终准确率
 val_accuracy = history.history['val_accuracy'][-1]
 results.append({'params': params, 'val_accuracy': val_accuracy})

# 输出结果
top_result = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
print("Best hyperparameter combination on CIFAR-10 is:", top_result)

