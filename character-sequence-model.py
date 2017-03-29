from scipy.misc import imread, imresize
#from keras.applications.imagenet_utils import preprocess_input
import os
import numpy as np
import matplotlib.pyplot as plt
import keras as ks
from keras.models import Model
from keras import models
from keras.layers import Flatten, Dropout, MaxPooling2D,Input
from keras.layers import Dense, InputLayer, Convolution2D
import logging
import numpy as np

import keras.backend as K
import tensorflow as tf


class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()
    """

    def __init__(self, input_tensor):
        self._build_network(input_tensor)

    def _build_network(self, input_tensor):
        
        model_input = Input(shape=(32,100,1))#,tensor=input_tensor)

        conv_layers_size = [64,128,256,512,512]
        edges_size = [5,5,3,3,3]
        
        #convolutional layes
        x = model_input 
        for i,(layer_size, edge_size) in enumerate(zip(conv_layers_size,edges_size)):
            x = Convolution2D(
                layer_size, edge_size, edge_size, 
                activation='relu', subsample=(1, 1),border_mode='same')(x)
            if i <= 2 :
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
        #dense layers
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096)(x)
    
    
        N = 23
        outputs = []
        for i in range(N):
            outputs.append(Dense(37,activation='softmax')(x))
        
        self.model = Model(input=model_input,output=outputs)

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model.output

    def __call__(self, input_tensor):
        return self.model(input_tensor)
    
    def tf_summary(self):
        print(self.model.summary())

    def save(self, filename):
        self.model.save(str(filename) + ".h5")
        print("Model saved to disk")


data = np.load("data.npy")
X = data[()]["X"]
y = data[()]["y"]
print(y)

model_cnn = CNN(tf.Variable(X))


model = model_cnn.model
opt = optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.fit(X, y,
          batch_size=64, nb_epoch=15, 
          validation_split=0.1,shuffle=True, verbose=2)

model_cnn.save('model1')

