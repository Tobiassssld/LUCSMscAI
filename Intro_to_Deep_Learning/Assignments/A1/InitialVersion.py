#!/usr/bin/env python
# coding: utf-8

# In first, we need to import the necessary libraries.

# In[6]:


from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import matplotlib.pyplot as plt


# Loading the MNIST & CIFAR-10 datasets. After loading, we need to normalize and one-hot encode the dataset. The process of loading could be found in the guidebook.

# In[7]:


# 加载 Fashion MNIST 数据集
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# 加载 CIFAR-10 数据集
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# 对 Fashion MNIST 进行归一化
x_train_fashion = x_train_fashion / 255.0
x_test_fashion = x_test_fashion / 255.0

# 对 CIFAR-10 进行归一化
x_train_cifar = x_train_cifar / 255.0
x_test_cifar = x_test_cifar / 255.0

# 对标签进行 one-hot 编码
y_train_fashion = to_categorical(y_train_fashion, 10)
y_test_fashion = to_categorical(y_test_fashion, 10)
y_train_cifar = to_categorical(y_train_cifar, 10)
y_test_cifar = to_categorical(y_test_cifar, 10)


# Construct the MLP model. First, flatten the 28x28 image into a one-dimensional array, then construct the first hidden layer (128 neurons, ReLU activation function) and the second hidden layer (64 neurons, ReLU activation function), and finally create the output layer (10 neurons, softmax activation function)

# In[8]:


# 构建 MLP 模型.神经元数量分别为128，64，10
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),   # 将 28x28 的图像展平为一维数组
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 输出层
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 创建MLP模型
mlp_model = create_mlp_model()
mlp_model.summary()

mlp_model.fit(x_train_fashion, y_train_fashion, epochs=25, validation_split=0.1, batch_size=32)


# When constructing the CNN model, we created two convolutional layers (with ReLU activation function and 64 and 32 filters respectively); used the MaxPooling2D() function for maximum pooling, and finally added a flatten layer to flatten the output of the convolutional layer into a one-dimensional array for use by the fully connected layer (Dense).

# In[9]:


# 构建 CNN 模型,其中卷积层中的滤波器数量分别为64,32
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Because Fashion MNIST is a grayscale image, the shape needs to be adjusted to (28, 28, 1). CIFAR-10 is a color image, and the input shape is (32, 32, 3).Save the training results for subsequent visualization

# In[ ]:


# 对 Fashion MNIST 使用 input_shape=(28, 28, 1)，CIFAR-10 使用 input_shape=(32, 32, 3)
cnn_model_fashion = create_cnn_model((28, 28, 1))
cnn_model_cifar = create_cnn_model((32, 32, 3))

# Fashion MNIST 需要在数据集维度上添加一维以匹配输入
x_train_fashion_cnn = x_train_fashion.reshape(-1, 28, 28, 1)
x_test_fashion_cnn = x_test_fashion.reshape(-1, 28, 28, 1)

cnn_model_fashion.fit(x_train_fashion_cnn, y_train_fashion, epochs=25, validation_split=0.1, batch_size=32)

# CIFAR-10 直接训练
cnn_model_cifar.fit(x_train_cifar, y_train_cifar, epochs=25, validation_split=0.1, batch_size=32)

# 训练模型并记录历史
history = mlp_model.fit(x_train_fashion, y_train_fashion, validation_data=(x_test_fashion, y_test_fashion), epochs=25)

# 将训练历史转换为 DataFrame进行可视化
history_df = pd.DataFrame(history.history)


# After converting the training history into a DataFrame, we can plot the results on the same line chart using the matplotlib library. Draw the chart after setting data labels for different parameters.

# In[ ]:


# 绘制训练历史折线图在同一张图上
plt.figure(figsize=(10, 6))

# 绘制 Training Loss 和 Validation Loss
plt.plot(history_df['loss'], label='Training Loss', linestyle='-', marker='o')
plt.plot(history_df['val_loss'], label='Validation Loss', linestyle='-', marker='o')

# 绘制 Training Accuracy 和 Validation Accuracy
plt.plot(history_df['accuracy'], label='Training Accuracy', linestyle='-', marker='s')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='s')

# 添加标题和标签
plt.title('Training and Validation Loss and Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

