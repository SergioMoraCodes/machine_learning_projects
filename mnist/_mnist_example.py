from pickletools import optimize
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from deeplearning_models import functional_model, MyCustomModel
from my_utils import display_examples

# tensorflow.keras.Sequential
seq_model= tf.keras.Sequential(
    [
        Input(shape=(28,28,1)), #size of the image, 1 represent the color chanel
        Conv2D(32, (3,3), activation='relu'), #[0] how many filters are [1] size of the filter
                                             #[2] 'relu'= rectified linear unit
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(), #computes average of the output values according to some axes
        Dense(64, activation='relu'), #vector containing values
        Dense(10, activation='softmax') #returns a set of 10 possible values, in propabilities

    ]
)
seq_model.summary()


if __name__=='__main__':


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Dataset visualization
    # print('x_train.shape = ', x_train.shape)
    # print('y_train.shape = ', y_train.shape)
    # print('x_test.shape = ', x_test.shape)
    # print('y_test.shape = ', y_test.shape)

    # display_examples(x_train,y_train)

    #normalazing the data, the gradient moves faster towards the global minimum of the cost function
    #it'a good practice but it's not necessary for every dataset
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    #transforming the dimension of the array, because input() takes one more dimension

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # to convert labels from normal encoding to one hot encoding, and use categorical_crossentropy
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)

    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    #hyperparameters to experiment, batch_size=amount of images that the model process at once
                                   #epochs=amount of times that the model goes trough all the data
                                   #validation_data=percentage of the total data to validate the model
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    model.evaluate(x_test,y_test, batch_size=64)
    model.summary()
