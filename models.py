#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:53:08 2022

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

This is a custom keras layer that converts light stimulus (R*/rod/s) into photoreceptor currents by using a biophysical model
of the photoreceptor by Rieke's lab "Predicting and Manipulating Cone Responses to Naturalistic Inputs. Juan M. Angueyra, Jacob Baudin, Gregory W. Schwartz, Fred Rieke
Journal of Neuroscience 16 February 2022, 42 (7) 1254-1274; DOI: 10.1523/JNEUROSCI.0793-21.2021


"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, MaxPool2D, BatchNormalization, GaussianNoise,LayerNormalization, PReLU
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import HeNormal



## CNN Model 1- ReLU activation function was used for convolutional layers 
def cnn2d(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    # Use an adaptive batch Normalization 
    
    
    y = LayerNormalization(axis=-3,epsilon=1e-7,trainable=False)(y)        # z-score the input across temporal dimension
    # y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            
        y = Activation('relu')(GaussianNoise(sigma)(y))
        

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))
        
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))
        

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    
    
    ############ Modified for C175 Final Project 
    #kernel_initializer=HeNormal(),
    
    # Softplus activation function 
    outputs = Activation('softplus')(y)
    
    # relu activation function 
    #outputs = Activation('relu')(y)
    
    # PRELU- adaptive (Parametric ReLU)
    # y=PReLU()(y)
    # outputs=y
    
    # GELU (Gaussian Error Linear Unit)
    #outputs = Activation('gelu')(y)
    
    # ELU (Exponential Linear Unit) 
    #outputs=Activation('elu')(y)
    
    

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)









## CNN Model #2 PReLU activation function was used for each convolutional layer 

def cnn2d2(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    # Use an adaptive batch Normalization 
    
    
    y = LayerNormalization(axis=-3,epsilon=1e-7,trainable=False)(y)        # z-score the input across temporal dimension
    # y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    # PReLU activation function 
    y=PReLU()(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            
        #y = Activation('relu')(GaussianNoise(sigma)(y))
        y=PReLU()(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        
        y=PReLU()(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        
        y=PReLU()(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    
    
    ############ Modified for C175 Final Project 
    #kernel_initializer=HeNormal(),
    # Softplus
    #outputs = Activation('softplus')(y)
    #outputs = Activation('relu')(y)
    
    # PRELU- adaptive (Parametric ReLU)
    y=PReLU()(y)
    outputs=y
    
    # GELU (Gaussian Error Linear Unit)
    #outputs = Activation('gelu')(y)
    
    # ELU (Exponential Linear Unit) 
    #outputs=Activation('elu')(y)
    
    

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)
