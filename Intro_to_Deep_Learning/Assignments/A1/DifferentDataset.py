#!/usr/bin/env python
# coding: utf-8

# In first, we need to import the necessary libraries.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Creating a Generic Model including 3 convolutional layers, output layer,dropout layer and dense layer.

# In[ ]:


# 通用模型创建函数
def create_model(input_shape, activation='relu', dropout_rate=0.2):
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

        Flatten(),
        Dense(256, activation=activation),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    return model


# Load the dataset and preprocess it, including normalizing and one-hot encoding the dataset.

# In[ ]:


# 数据集加载和预处理
def load_and_preprocess_data(dataset):
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    else:
        raise ValueError("Unknown dataset")

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# We create the evaluate_model_on_dataset() function to load and preprocess the dataset, create and compile the model, train the model and return the training history

# In[ ]:


# 模型评估并返回历史的函数
def evaluate_model_on_dataset(model, dataset_name, epochs=10, batch_size=32):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(dataset_name)
    input_shape = x_train.shape[1:]  # 自动适应数据集的输入形状

    # 创建模型
    model = create_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=batch_size, verbose=1)

    # 返回训练历史
    return pd.DataFrame(history.history)


# 在 Fashion MNIST 和 CIFAR-10 上测试模型，并记录历史
print("Training on Fashion MNIST:")
fashion_mnist_history = evaluate_model_on_dataset(None, 'fashion_mnist', epochs=20)

print("\nTraining on CIFAR-10:")
cifar10_history = evaluate_model_on_dataset(None, 'cifar10', epochs=20)


# Finally, we call the matplotlib library to visualize the training results to compare the performance of the same model on two different databases.

# In[ ]:


# 可视化训练结果
#put the curves in the same graph to compare the results
plt.figure(figsize=(12, 8))

# 绘制 Fashion MNIST 的训练曲线
plt.plot(fashion_mnist_history['accuracy'], label='Fashion MNIST Training Accuracy', color='blue')
plt.plot(fashion_mnist_history['val_accuracy'], label='Fashion MNIST Validation Accuracy', color='blue', linestyle='--')
plt.plot(fashion_mnist_history['loss'], label='Fashion MNIST Training Loss', color='orange')
plt.plot(fashion_mnist_history['val_loss'], label='Fashion MNIST Validation Loss', color='orange', linestyle='--')

# 绘制 CIFAR-10 的训练曲线
plt.plot(cifar10_history['accuracy'], label='CIFAR-10 Training Accuracy', color='green')
plt.plot(cifar10_history['val_accuracy'], label='CIFAR-10 Validation Accuracy', color='green', linestyle='--')
plt.plot(cifar10_history['loss'], label='CIFAR-10 Training Loss', color='red')
plt.plot(cifar10_history['val_loss'], label='CIFAR-10 Validation Loss', color='red', linestyle='--')

# 设置图表标题和标签
plt.title("Training and Validation Metrics for Fashion MNIST and CIFAR-10")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

