import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D



# tensorflow.keras.Sequential

model= tf.keras.Sequential(
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

#functional approach : function that returns a model

# tensorflow.keras.Model : inherit from this class and modify the object

def display_examples(examples, labels):
    plt.figure(figsize=(8,8))
    for i in range(25):
        idx = np.random.randint(0,examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()
if __name__=='__main__':
    
    #Dataset visualization
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    display_examples(x_train,y_train)