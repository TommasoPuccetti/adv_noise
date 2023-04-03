import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import tensorflow as tf
from scipy.spatial.distance import pdist, cdist, squareform
from keras import regularizers
from sklearn.decomposition import PCA   

def get_model(softmax=False):
    
    layers = [
        Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
        Activation('relu'),  # 1
        BatchNormalization(), # 2
        Conv2D(32, (3, 3), padding='same'),  # 3
        Activation('relu'),  # 4
        BatchNormalization(), # 5
        MaxPooling2D(pool_size=(2, 2)),  # 6
            
        Conv2D(64, (3, 3), padding='same'),  # 7
        Activation('relu'),  # 8
        BatchNormalization(), # 9
        Conv2D(64, (3, 3), padding='same'),  # 10
        Activation('relu'),  # 11
        BatchNormalization(), # 12
        MaxPooling2D(pool_size=(2, 2)),  # 13
            
        Conv2D(128, (3, 3), padding='same'),  # 14
        Activation('relu'),  # 15
        BatchNormalization(), # 16
        Conv2D(128, (3, 3), padding='same'),  # 17
        Activation('relu'),  # 18
        BatchNormalization(), # 19
        MaxPooling2D(pool_size=(2, 2)),  # 20
            
        Flatten(),  # 21
        Dropout(0.5),  # 22
            
        Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 23
        Activation('relu'),  # 24
        BatchNormalization(), # 25
        Dropout(0.5),  # 26
        Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 27
        Activation('relu'),  # 28
        BatchNormalization(), # 29
        Dropout(0.5),  # 30
        Dense(10),  # 31

    ]

    model = Sequential()

    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))

    return model