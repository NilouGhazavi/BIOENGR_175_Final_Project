#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

"""
## GPU
#!pip install tensorflow
import tensorflow as tf

# List all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# If GPUs are available, set the desired one to be visible to TensorFlow
if gpus:
    try:
        # Specify the GPU(s) to be used
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU for the first model
        # For the second model, you might run it in a separate script or later in the same script
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # Use the second GPU
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


import matplotlib.pyplot as plt
import numpy as np
import os
import models
from models import *
from data_handler import prepare_data_cnn2d, load_h5Dataset, model_evaluate_new
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
import gc
print(tf.__version__)
tf.compat.v1.disable_eager_execution()


    
# load train val and test datasets from saved h5 file
"""
    load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    and data_train.y contains the spikerate normalized by median [samples,numOfCells]
"""


fname_data_train_val_test_all = 'monkey_data/monkey01_dataset_train_val_test_scot-30-Rstar.h5'


idx_train_start = 0    # mins to chop off in the begining.
trainingSamps_dur = 30      # minutes (total is like 50 mins i think. If you put -1 here then you will load all the data
validationSamps_dur = 1 # minutes
testSamps_dur = 0.3 # minutes

rgb = load_h5Dataset(fname_data_train_val_test_all,nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                     idx_train_start=idx_train_start)

#START HERE 
data_train=rgb[0]
data_val = rgb[1]
data_test = rgb[2]
data_quality = rgb[3]
dataset_rr = rgb[4]
parameters = rgb[5]
if len(rgb)>7:
    data_info = rgb[7]

t_frame = parameters['t_frame']     # time in ms of one frame/sample 
"""
data_train.X contains the movie frames [time,y,x]
data_train.y contains the firing rate [time,rgc]

Same for data_val and data_test. The three datasets (train, val and test) will not overlap

"""

# %% Arrange data in training samples format. Roll the time dimension to have movie chunks equal to the temporal width of your CNN
temporal_width = 120   # frames (120 frames ~ 1s)
idx_unitsToTake = np.arange(0,data_train.y.shape[-1])   # Take all the rgcs
data_train = prepare_data_cnn2d(data_train,temporal_width,idx_unitsToTake)     # [samples,temporal_width,rows,columns]
data_test = prepare_data_cnn2d(data_test,temporal_width,idx_unitsToTake)
data_val = prepare_data_cnn2d(data_val,temporal_width,idx_unitsToTake)   





#%%% Data Visualization 
# Code to generate Figure 6 in the report 
## Modified for BIOENGR175 Final Project 


# Train Data
print(f"Shape of the train data is: {data_train[0].shape}")
print(f"Shape of the validation data is: {data_val[1].shape}")
print(f"Shape of the testn data is: {data_test[1].shape}")

training=rgb[0]
movie_frames_t=data_train.X
firing_rate_t=data_train.y
num_samples_train=movie_frames_t.shape[0]
num_frames_per_sample=movie_frames_t.shape[1]

print(f"Shape of train dataset_movie frames {movie_frames_t.shape} and firing rate is {firing_rate_t.shape}")
print(f"there are {num_samples_train} samples, each sample consists of {num_frames_per_sample} frames ")


# there are 120 frames per sample 
plt.imshow(movie_frames_t[1,1,:,:],cmap='gray')
plt.imshow(movie_frames_t[1,2,:,:],cmap='gray')
plt.imshow(movie_frames_t[1,119,:,:],cmap='gray')


# there is one firing rate per sample 
plt.figure(figsize=(20,15))
plt.plot(firing_rate_t[:,1])





# # Validation data 
validation=rgb[1]
movie_frames_v=data_val.X
firing_rate_v=data_val.y
num_samples_val=movie_frames_v.shape[0]
num_frames_per_sample_val=movie_frames_v.shape[1]
print(f"Shape of validation dataset_movie frames {movie_frames_v.shape} and firing rate is {firing_rate_v.shape}")
print(f"there are {num_samples_val} samples, each sample consists of {num_frames_per_sample_val} frames ")



plt.imshow(movie_frames_v[1,1,:,:],cmap='gray')
plt.imshow(movie_frames_v[1,2,:,:],cmap='gray')
plt.imshow(movie_frames_v[1,119,:,:],cmap='gray')

# there is one firing rate per sample 
plt.figure(figsize=(20,15))
plt.plot(firing_rate_v[:,1])




# Test data 
#testing=rgb[2]
movie_frames_te=data_test.X
firing_rate_te=data_test.y
num_samples_te=movie_frames_te.shape[0]
num_frames_per_sample_te=movie_frames_te.shape[1]
print(f"Shape of testing dataset_movie frames {movie_frames_te.shape} and firing rate is {firing_rate_te.shape}")
print(f"there are {num_samples_te} samples, each sample consists of {num_frames_per_sample_te} frames ")



plt.imshow(movie_frames_te[1,1,:,:],cmap='gray')
plt.imshow(movie_frames_te[1,2,:,:],cmap='gray')
plt.imshow(movie_frames_te[1,119,:,:],cmap='gray')

# there is one firing rate per sample 
plt.figure(figsize=(20,15))
plt.plot(firing_rate_te[:,1])






# %% Build a conventional-CNN

chan1_n=8
filt1_size=9
chan2_n=16
filt2_size=7
chan3_n=18
filt3_size=5
bz=125
BatchNorm=1
MaxPool=1



dict_params = dict(chan1_n=chan1_n,filt1_size=filt1_size,
                   chan2_n=chan2_n,filt2_size=filt2_size,
                   chan3_n=chan3_n,filt3_size=filt3_size,
                   filt_temporal_width=temporal_width,
                   BatchNorm=BatchNorm,MaxPool=MaxPool,
                   dtype='float32')

n_rgcs = data_train.y.shape[-1]        # number of units in output layer
inp_shape = Input(shape=data_train.X.shape[1:]) # keras input layer




mdl = models.cnn2d(inp_shape,n_rgcs,**dict_params)
mdl.summary()





# %% Train model
lr = 0.0001
nb_epochs=100

mdl.compile(loss='poisson', optimizer=tf.keras.optimizers.legacy.Adam(lr))
mdl_history = mdl.fit(x=data_train.X, y=data_train.y, validation_data=(data_val.X,data_val.y), validation_freq=1, shuffle=True,batch_size=bz, epochs=nb_epochs)

                                                                                                                                                                                                                                                                                                            
mdl.save('/Trained_Model_monkey01_dataset_scot-30-Rstar/trained_model_30.h5')
# Load the model 
mdl=load_model(r"/Trained_Model_monkey01_dataset_scot-30-Rstar/trained_model_30.h5")




# %% Model Evaluation

## Code to generate figures 8-11 in the report 

# training loss and validation loss
train_loss=mdl_history.history['loss']
val_loss=mdl_history.history['val_loss']
epochs=range(1,len(train_loss)+1)
plt.plot(epochs,train_loss)
plt.plot(epochs,val_loss)
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss','Validation Loss'])



# Fraction of Explained Variance 
fev_allUnits = np.zeros((nb_epochs,n_rgcs))
fev_allUnits[:] = np.nan



# Prediction Error 
predCorr_allUnits = np.zeros((nb_epochs,n_rgcs))
predCorr_allUnits[:] = np.nan

# for compatibility with greg's dataset


obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
obs_noise = None
num_iters = 10


samps_shift = 0 

# Validation dataset or test dataset 
#pred_rate = mdl.predict(data_test.X)
pred_rate = mdl.predict(data_val.X)


fev_loop = np.zeros((num_iters,n_rgcs))
fracExVar_loop = np.zeros((num_iters,n_rgcs))
predCorr_loop = np.zeros((num_iters,n_rgcs))
rrCorr_loop = np.zeros((num_iters,n_rgcs))

for j in range(num_iters):  # nunm_iters is 1 with my dataset. This was mainly for greg's data where we would randomly split the dataset to calculate performance metrics 
    fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,filt_width=temporal_width,lag=int(samps_shift),obs_noise=obs_noise)
    
fev = np.mean(fev_loop,axis=0)


fev_allUnits = np.mean(fev_loop,axis=0)
fev_medianUnits = np.nanmedian(fev_allUnits)      


num_RGC=37
plt.plot(range(num_RGC),fev_allUnits)
#plt.axhline(y=fev_medianUnits(), color='r',linestyle='--', label='Median FEV')
plt.xlabel('Retinal Ganglion Cell')
plt.ylabel('Fraction of Explained Variance')




predCorr_allUnits = np.mean(predCorr_loop,axis=0)
predCorr_medianUnits = np.nanmedian(predCorr_allUnits)


num_RGC=37
plt.scatter(range(num_RGC),predCorr_allUnits )
#plt.axhline(y=predCorr_medianUnits, color='r',linestyle='--', label='Median FEV')
plt.xlabel('Retinal Ganglion Cell')
plt.ylabel('Prediction Correlation for All RGCs')

_ = gc.collect()



plt.plot(data_val.y[:,19]); plt.plot(pred_rate[:,19])
plt.legend(['RGC actual response','RGC predicted response'])

#END HERE 
# trained model 

#mdl.save('/Trained_Model_monkey01_dataset_scot-30-Rstar/trained_model_30.h5')



"""
#%% Cross Validation 
# Code to generate cross validation results

# *****************  Modified for 175 final project 
from tensorflow.keras.utils import Sequence 
from sklearn.model_selection import KFold 


## Cross Validation with the training dataset 

## Loading the entire dataset--> Memory error (batch size=32)
class DataGenerator(Sequence):
    def __init__(self, data_X, data_y, indices, batch_size=32):
        self.data_X = data_X
        self.data_y = data_y
        self.indices = indices
        self.batch_size = batch_size
    def __len__(self):
        return np.ceil(len(self.indices) / self.batch_size).astype(int)
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.data_X[batch_indices]
        batch_y = self.data_y[batch_indices]
        return np.array(batch_X), np.array(batch_y)
    
    
# number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# Only train dataset  (train.X.shape: (224881,120,30,39), val.X.shape: (1072,120,30,39))
data_X=data_train.X
data_y=data_train.y
fold_results = []
for fold, (train_indices, test_indices) in enumerate(kf.split(data_X)):
    print(f"Processing fold {fold+1}")
    
    train_generator = DataGenerator(data_X, data_y, train_indices, batch_size=32)
    test_generator = DataGenerator(data_X, data_y, test_indices, batch_size=32)
    
    inp_shape = Input(shape=data_X.shape[1:])
    model = models.cnn2d(inp_shape, n_rgcs, **dict_params)
    model.compile(loss='poisson', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # Train the model
    model.fit(train_generator, epochs=100, verbose=1)
    # Evaluate the model/ loss value
    evaluation=  model.evaluate(test_generator)
    # Store fold results
    fold_results.append((evaluation))
    print(f"Fold {fold+1} -  loss: {evaluation}")
    # Save each model
    model_path = f'/model_fold_{fold+1}.h5'
    model.save(model_path)
    print(f'Model for fold {fold+1} saved to {model_path}')
"""





