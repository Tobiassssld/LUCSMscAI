#!/usr/bin/env python
# coding: utf-8

# In first, we need to import the necessary libraries.
# 
# 

# In[ ]:


from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import matplotlib.pyplot as plt


# Loading the MNIST & CIFAR-10 datasets. After loading, we need to normalize and one-hot encode the dataset. The process of loading will be a little slow.

# In[ ]:


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


# Now let’s build the MLP model. After initially building the MLP model, we upgraded its network architecture, including adding a Dropout layer and increasing the number of neurons. We also tested different optimizers.

# In[ ]:


# 构建 MLP 模型
def create_improved_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dropout(0.5),  # 新增 Dropout 层
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.0005),  # 改用 RMSprop 优化器
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),  # 改用 SGD 优化器 Change optimizer as SGD
                  #loss='categorical_crossentropy', metrics=['accuracy'])
    return model

improved_model = create_improved_model()
improved_model.fit(x_train_fashion, y_train_fashion, epochs=20, validation_split=0.1, batch_size=32)


# When building the CNN model, we also upgraded the structure compared to the initial version, including increasing the number of convolutional layers and filters.We also tested different optimizers to see the different results of the accuracy.

# In[ ]:


# 构建 CNN 模型
def create_deeper_cnn_model(input_shape):
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # 展平并连接全连接层
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.0005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),  # 改用 SGD 优化器 Change optimizer as SGD
                 #loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# After adjusting the parameters, continue training the two datasets. Save the training results for subsequent visualization.:

# In[ ]:


# 对 Fashion MNIST 使用 input_shape=(28, 28, 1)，对 CIFAR-10 使用 input_shape=(32, 32, 3)
cnn_model_fashion = create_deeper_cnn_model((28, 28, 1))
cnn_model_cifar = create_deeper_cnn_model((32, 32, 3))

# 训练 Fashion MNIST 模型
x_train_fashion_cnn = x_train_fashion.reshape(-1, 28, 28, 1)
cnn_model_fashion.fit(x_train_fashion_cnn, y_train_fashion, epochs=20, validation_split=0.1, batch_size=32)

# 训练 CIFAR-10 模型
cnn_model_cifar.fit(x_train_cifar, y_train_cifar, epochs=20, validation_split=0.1, batch_size=32)

# 训练模型并记录历史
history = improved_model.fit(x_train_fashion, y_train_fashion, validation_data=(x_test_fashion, y_test_fashion), epochs=20)


# After converting the training history into a DataFrame, we can plot the results on the same line chart using the matplotlib library. Draw the chart after setting data labels for different parameters.

# In[ ]:


# 将训练历史转换为 DataFrame
history_df = pd.DataFrame(history.history)

# 绘制训练历史折线图在同一张图上
plt.figure(figsize=(10, 6))

# 绘制 Training Loss 和 Validation Loss
plt.plot(history_df['loss'], label='Training Loss', linestyle='-', marker='o')
plt.plot(history_df['val_loss'], label='Validation Loss', linestyle='-', marker='o')

# 绘制 Training Accuracy 和 Validation Accuracy
plt.plot(history_df['accuracy'], label='Training Accuracy', linestyle='-', marker='o')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='o')

# 添加标题和标签
plt.title('Training and Validation Loss and Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

