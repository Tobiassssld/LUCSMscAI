# %% [markdown]
# <div style="text-align: right">   </div>
# 
# 
# Introduction to Deep Learning (2024) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| &nbsp;
# -------|-------------------
# **Assignment 2 - Sequence processing using RNNs** | <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/UniversiteitLeidenLogo.svg/1280px-UniversiteitLeidenLogo.svg.png" width="300">
# 
# 
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description**
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, BatchNormalization, Dropout
from tensorflow.keras.layers import RepeatVector, Conv2D, Conv3D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose, Flatten, Input, MaxPooling2D, MaxPooling3D
import keras

import matplotlib.pyplot as plt

# %%
seed = 8032
keras.utils.set_random_seed(seed)

# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
        if sign == '*':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            # Rotate 45 degrees
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %% [markdown]
# ## Helper functions
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

# %%
def plotter_1(X, y_true, text_preds):
    error_counter = 0
    error_in = np.zeros(13, dtype=np.int16)
    
    for i in range(text_preds.shape[0]):
        if decode_labels(y_true[i]) != decode_labels(text_preds[i]):
            error_in += np.where(np.sum(X[i], axis=0)>0, 1, 0)
            error_counter += 1

    x = np.arange(10)
    width = 0.25
    chars = [str(i) for i in range(10)]
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.bar(x, error_in[:10])
    plt.xticks(x, chars)
    plt.title("Error Count per Number in Query")
    
    plt.subplot(1,2,2)
    plt.pie(error_in[10:12], labels=[unique_characters[i] for i in range(10, 12)], autopct='%1.1f%%', pctdistance=1.25, labeldistance=.6)
    plt.title("Percentage of Mistakes by Operation Sign")
    
    plt.show()
    return error_counter, error_in

# %%
def plotter_2(y_true, text_preds):
    error_counter = 0
    error_out = np.zeros(13, dtype=np.int16)
    pos_err = 0
    
    for i in range(text_preds.shape[0]):
        if decode_labels(y_true[i]) != decode_labels(text_preds[i]):
            error_out += np.where(np.sum(y_true[i], axis=0)>0, 1, 0)
            error_counter += 1
            if np.sum(y_true[i], axis=0)[11] == 0:
                pos_err += 1
            
    x = np.arange(10)
    width = 0.25
    chars = [str(i) for i in range(10)]
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.bar(x, error_out[:10])
    plt.xticks(x, chars)
    plt.title("Error Count per Number in True Output")

    plt.subplot(1,2,2)
    plt.pie([pos_err,error_out[11]], labels=["+","-"], autopct='%1.1f%%', pctdistance=1.25, labeldistance=.6)
    plt.title("Percentage of mistakes by Output sign")
    plt.show()
    return error_counter, error_out

# %%
def mean_absolute_error(y_true,y_pred):
    err = 0

    for i in range(y_true.shape[0]):
        y = decode_labels(y_true[i])
        yhat = decode_labels(y_pred[i])
        try:
            err += abs(int(y)-int(yhat))
        except ValueError:
            err += int(y)
            
    
    return err/y_true.shape[0]

# %% [markdown]
# # Models

# %% [markdown]
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %% [markdown]
# ### Test Size 0.5

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text_onehot, y_text_onehot, test_size=0.5)

t2t_model = build_text2text_model()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())


t2t_model.fit(X_train_txt, y_train_txt, epochs=20)

# %%
print(t2t_model.evaluate(X_test_txt, y_test_txt))

# %%
t2t_preds = t2t_model.predict(X_test_txt)
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, t2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, t2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
mean_absolute_error(y_test_txt, t2t_preds)

# %% [markdown]
# ### Test Size 0.75

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text_onehot, y_text_onehot, test_size=0.75)

t2t_model = build_text2text_model()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())


t2t_model.fit(X_train_txt, y_train_txt, epochs=20)

# %%
print(t2t_model.evaluate(X_test_txt, y_test_txt))

# %%
t2t_preds = t2t_model.predict(X_test_txt)
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, t2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, t2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, t2t_preds))

# %% [markdown]
# ### Test Size 0.9

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text_onehot, y_text_onehot, test_size=0.9)

t2t_model = build_text2text_model()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())


t2t_model.fit(X_train_txt, y_train_txt, epochs=20)

# %%
print(t2t_model.evaluate(X_test_txt, y_test_txt))

# %%
t2t_preds = t2t_model.predict(X_test_txt)
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, t2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, t2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, t2t_preds))

# %% [markdown]
# ### Extra LSTM Layers in Encoder

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, return_sequences=True, input_shape=(None, len(unique_characters))))
    text2text.add(LSTM(256, return_sequences=True))
    text2text.add(LSTM(256))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text_onehot, y_text_onehot, test_size=0.75)

t2t_model = build_text2text_model()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())


t2t_model.fit(X_train_txt, y_train_txt, epochs=20)

# %%
print(t2t_model.evaluate(X_test_txt, y_test_txt))

# %%
t2t_preds = t2t_model.predict(X_test_txt)
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, t2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, t2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, t2t_preds))

# %% [markdown]
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %%
## Your code
def build_img2text_model_flat():

    # We start by initializing a sequential model
    img2text = tf.keras.Sequential()
    img2text.add(Input(shape=(max_query_length, *MNIST_data.shape[1:])))
    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    img2text.add(Reshape((5, 28*28)))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, activation='relu'))
    img2text.add(Dropout(0.5))
    #img2text.add(LSTM(256))
    img2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    img2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img2text.summary()

    return img2text




# %%
## Your code
def build_img2text_model_convlstm():

    # We start by initializing a sequential model
    img2text = tf.keras.Sequential()
    img2text.add(Input(shape=(max_query_length, *MNIST_data.shape[1:], 1)))
    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    img2text.add(ConvLSTM2D(32, (3,3), padding="same", activation='relu', return_sequences=True))
    img2text.add(BatchNormalization())
    img2text.add(MaxPooling3D((1,2,2)))
    img2text.add(ConvLSTM2D(64, (3,3), padding="same", activation='relu', return_sequences=True))
    img2text.add(MaxPooling3D((1,2,2)))
    img2text.add(BatchNormalization())
    img2text.add(ConvLSTM2D(128, (3,3), padding="same", activation='relu'))
    img2text.add(MaxPooling2D((2,2)))
    img2text.add(BatchNormalization())
    img2text.add(Flatten())
    
    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    img2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    img2text.add(LSTM(512, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(512, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(512, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(512, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(512, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    img2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img2text.summary()

    return img2text




# %% [markdown]
# ### Training and results

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text)
X_train_img, X_test_img, y_train_txt, y_test_txt, X_train_txt, X_test_txt = train_test_split(X_img, y_text_onehot, X_text_onehot, test_size=0.2)
X_train_img, X_val_img, y_train_txt, y_val_txt = train_test_split(X_train_img, y_train_txt, test_size=0.2)

i2t_model = build_img2text_model_flat()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
hist = i2t_model.fit(X_train_img, y_train_txt, validation_data=(X_val_img, y_val_txt), epochs=100, callbacks=[es_callback])

# %%
print(i2t_model.evaluate(X_test_img, y_test_txt))

# %%
i2t_preds = i2t_model.predict(X_test_img)
for i in range(i2t_preds.shape[0]):
    for t in range(3):
        pred = np.argmax(i2t_preds[i,t,:])
        i2t_preds[i,t,:] *= 0
        i2t_preds[i,t,pred] = 1
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, i2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, i2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, i2t_preds))

# %%
plt.plot(hist.history["val_accuracy"])

# %%
i2t_model = build_img2text_model_convlstm()

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
hist = i2t_model.fit(X_train_img, y_train_txt, validation_data=(X_val_img, y_val_txt), epochs=100, callbacks=[es_callback])

# %%
print(i2t_model.evaluate(X_test_img, y_test_txt))

# %%
i2t_preds = i2t_model.predict(X_test_img)
for i in range(i2t_preds.shape[0]):
    for t in range(3):
        pred = np.argmax(i2t_preds[i,t,:])
        i2t_preds[i,t,:] *= 0
        i2t_preds[i,t,pred] = 1
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, i2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, i2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, i2t_preds))

# %%
plt.plot(hist.history["val_accuracy"])

# %% [markdown]
# ### Extra LSTM in Encoder

# %%
## Your code
def build_img2text_model_flat_extra():

    # We start by initializing a sequential model
    img2text = tf.keras.Sequential()
    img2text.add(Input(shape=(max_query_length, *MNIST_data.shape[1:])))
    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    img2text.add(Reshape((5, 28*28)))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, return_sequences=True, activation='relu'))
    img2text.add(Dropout(0.5))
    img2text.add(LSTM(256, activation='relu'))
    img2text.add(Dropout(0.5))
    #img2text.add(LSTM(256))
    img2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    img2text.add(LSTM(512, return_sequences=True))
    img2text.add(Dropout(0.5))
    img2text.add(BatchNormalization())
    

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    img2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img2text.summary()

    return img2text




# %%
i2t_model = build_img2text_model_flat_extra()
#tensorboard_cb = keras.callbacks.TensorBoard(log_root())
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
hist = i2t_model.fit(X_train_img, y_train_txt, validation_data=(X_val_img, y_val_txt), epochs=100, callbacks=[es_callback])

# %%
print(i2t_model.evaluate(X_test_img, y_test_txt))

# %%
i2t_preds = i2t_model.predict(X_test_img)
for i in range(i2t_preds.shape[0]):
    for t in range(3):
        pred = np.argmax(i2t_preds[i,t,:])
        i2t_preds[i,t,:] *= 0
        i2t_preds[i,t,pred] = 1
error_counter, error_in = plotter_1(X_test_txt, y_test_txt, i2t_preds)

# %%
_, error_out = plotter_2(y_test_txt, i2t_preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, i2t_preds))

# %%
plt.plot(hist.history["val_accuracy"])

# %% [markdown]
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %%
# Your code
def build_text2img_model():

    # We start by initializing a sequential model
    txt2img = tf.keras.Sequential()
    txt2img.add(LSTM(256, return_sequences=True, input_shape=(None, len(unique_characters)), activation='relu'))
    txt2img.add(LSTM(256, return_sequences=True, activation='relu'))
    txt2img.add(LSTM(256, activation='relu'))

    txt2img.add(RepeatVector(max_answer_length))

    txt2img.add(LSTM(784, return_sequences=True, activation='relu'))#TODO add more lstm layers here (might need remove a 2d transpose, unsure)
    txt2img.add(Reshape((3, 28, 28)))

    txt2img.add(Conv2DTranspose(128, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(64, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(32, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(3, (3,3), padding="same", data_format="channels_first", activation='sigmoid'))

    #txt2img.add(Reshape((3, 28, 28)))

    txt2img.compile(loss="mse", optimizer='adam', metrics=['mean_absolute_error'])
    txt2img.summary()
    
    return txt2img



# %%
# Your code
def build_text2img_model_ExtraLstm():

    # We start by initializing a sequential model
    txt2img = tf.keras.Sequential()
    txt2img.add(LSTM(256, return_sequences=True, input_shape=(None, len(unique_characters)), activation='relu'))
    txt2img.add(LSTM(256, return_sequences=True, activation='relu'))
    txt2img.add(LSTM(256, return_sequences=True, activation='relu'))
    txt2img.add(LSTM(256, return_sequences=True, activation='relu'))
    txt2img.add(LSTM(256, return_sequences=True, activation='relu'))
    txt2img.add(LSTM(256, activation='relu'))

    txt2img.add(RepeatVector(max_answer_length))

    txt2img.add(LSTM(784, return_sequences=True, activation='relu'))#TODO add more lstm layers here (might need remove a 2d transpose, unsure)
    txt2img.add(Reshape((3, 28, 28)))

    txt2img.add(Conv2DTranspose(128, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(64, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(32, (3,3), padding="same", data_format="channels_first", activation='relu'))
    txt2img.add(Conv2DTranspose(3, (3,3), padding="same", data_format="channels_first", activation='sigmoid'))

    #txt2img.add(Reshape((3, 28, 28)))

    txt2img.compile(loss="mse", optimizer='adam', metrics=['mean_absolute_error'])
    txt2img.summary()
    
    return txt2img



# %%
X_train_txt, X_test_txt, y_train_img, y_test_img, y_train_txt, y_test_txt = train_test_split(X_text_onehot, y_img, y_text_onehot, test_size=0.2)
X_train_txt, X_val_txt, y_train_img, y_val_img = train_test_split(X_train_txt, y_train_img, test_size=0.2)

t2i_model = build_text2img_model()

t2i_model.fit(X_train_txt, y_train_img, validation_data=(X_val_txt, y_val_img), epochs=50)

# %%
t2i_model_extra = build_text2img_model_ExtraLstm()

t2i_model_extra.fit(X_train_txt, y_train_img, validation_data=(X_val_txt, y_val_img), epochs=50)

# %% [markdown]
# ### Some output examples

# %%
plt.imshow(np.hstack((y_train_img[0,0], y_train_img[0,1], y_train_img[0,2])), cmap="gray")

# %%
yhat = t2i_model.predict(X_train_txt[:2,:])[0,:]
print(yhat.shape)
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
yhat = t2i_model_extra.predict(X_train_txt[:2,:])[0,:]
print(yhat.shape)
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
plt.imshow(np.hstack((y_train_img[1,0], y_train_img[1,1], y_train_img[1,2])), cmap="gray")

# %%
yhat = t2i_model.predict(X_train_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
yhat = t2i_model_extra.predict(X_train_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
plt.imshow(np.hstack((y_val_img[1,0], y_val_img[1,1], y_val_img[1,2])), cmap="gray")

# %%
yhat = t2i_model.predict(X_val_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
yhat = t2i_model_extra.predict(X_val_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
plt.imshow(np.hstack((y_test_img[1,0], y_test_img[1,1], y_test_img[1,2])), cmap="gray")

# %%
yhat = t2i_model.predict(X_test_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %%
yhat = t2i_model_extra.predict(X_test_txt[:2,:])[1,:]
plt.imshow(np.hstack((yhat[0,:], yhat[1,:], yhat[2,:])), cmap="gray")

# %% [markdown]
# ### Generated Image Classification Model

# %%
def build_discrim_model():

    # We start by initializing a sequential model
    discrim = tf.keras.Sequential()

    discrim.add(Input(shape=(3,28,28)))
    discrim.add(Reshape((3, 28 * 28)))
    discrim.add(LSTM(256, return_sequences=True, activation='relu'))
    discrim.add(LSTM(256, return_sequences=True, activation='relu'))
    discrim.add(LSTM(256, return_sequences=True, activation='relu'))

    discrim.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    discrim.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    discrim.summary()
    
    return discrim

# %%
X_train_img, X_test_img, y_train_txt, y_test_txt = train_test_split(y_img, y_text_onehot, test_size=0.2)
discrim_model = build_discrim_model()

discrim_model.fit(X_train_img, y_train_txt, epochs=20)

# %%
print(discrim_model.evaluate(X_test_img, y_test_txt))

# %%
print(mean_absolute_error(y_test_txt, discrim_model.predict(X_test_img)))

# %% [markdown]
# ### Text To Image Model Evaluation

# %%
discrim_model.fit(y_img, y_text_onehot, epochs=20, callbacks=[es_callback])

# %%
img_gen = t2i_model.predict(X_test_txt)
preds = discrim_model.predict(img_gen)
for i in range(preds.shape[0]):
    for t in range(3):
        pred = np.argmax(preds[i,t,:])
        preds[i,t,:] *= 0
        preds[i,t,pred] = 1

error_counter, error_in = plotter_1(X_test_txt, y_test_txt, preds)

# %%
_, error_out = plotter_2(y_test_txt, preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, preds))

# %%
print(1-(error_counter / y_test_txt.shape[0]))

# %%
img_gen = t2i_model_extra.predict(X_test_txt)
preds = discrim_model.predict(img_gen)
for i in range(preds.shape[0]):
    for t in range(3):
        pred = np.argmax(preds[i,t,:])
        preds[i,t,:] *= 0
        preds[i,t,pred] = 1

error_counter, error_in = plotter_1(X_test_txt, y_test_txt, preds)

# %%
_, error_out = plotter_2(y_test_txt, preds)

# %%
print(error_counter)
print(error_in)
print(error_out)

# %%
print(mean_absolute_error(y_test_txt, preds))

# %%
print(1-(error_counter / y_test_txt.shape[0]))


