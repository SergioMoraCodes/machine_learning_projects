import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold(a):
    return np.piecewise(a, [a<0, a>=0],[0,1])

def signum(a):
    return np.piecewise(a, [a<0, a>=0],[-1,1])

def relu(a):
    #np.piecewise(a,[a<0, a>=0], [0,lambda a: a])
    return np.maximum(0,a)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

if __name__=='__main__':

    x = np.linspace(10,-10,120)

    plt.plot(x,softmax(x))