#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:23:32 2021

@author: saad
""" 

import numpy as np
import os
from scipy import io
import re
import h5py
import math
import matplotlib.pyplot as plt
from global_scripts import spiketools

def applyLightIntensities(meanIntensity,data,t_frame):

    X = data
    X = X.astype('float32')
    idx_low = X<0.5
    idx_high = X>0.5
    X[idx_high] = 2*meanIntensity
    X[idx_low] = (2*meanIntensity)/300
    
    X = X * 1e-3 * t_frame  # photons per time bin 

        
    data = X
    return data

def normalizeLightIntensities(data):
    
    f = h5py.File(fname_dataFile,'r')
    data_normAgainst = f['scot-30']['train']['stim_frames'][:1000]
    norm_mean =np.mean(data_normAgainst)
    norm_std = np.std(data_normAgainst)
    
    X = data.X
    X = X.astype('float32')
    X = (X-X.mean())/norm_std
    
    data = Exptdata(X,data.y)
    return data
 

expDate = 'monkey01'
UP_SAMP = 0
bin_width = 8   # ms

CONVERT_RSTAR = True
NORM_STIM = 1


path_raw = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten',expDate,'db_files/raw')
path_db = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten',expDate,'db_files/datasets')
path_save = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'db_files/datasets')
if CONVERT_RSTAR==True:
    fname_save = os.path.join(path_save,(expDate+'_dataset_CB_'+'allLightLevels'+'_'+str(bin_width)+'ms'+'_Rstar.h5'))
else:
    fname_save = os.path.join(path_save,(expDate+'_dataset_CB_'+'allLightLevels'+'_'+str(bin_width)+'ms.h5'))


if not os.path.exists(path_save):
    os.mkdir(path_save)


matDataFile = os.path.join(path_raw,'saad_structure'+'.mat')
matData = io.loadmat(matDataFile)
matData = np.array(matData['saad_structure'])

cellindex = os.path.join(path_raw,'cell_index_struct'+'.mat')
cellindex = io.loadmat(cellindex)
cellindex = cellindex['cell_index_struct']

idx_on_par = cellindex['on_parasol_indices'][0][0] - 1
idx_on_mid = cellindex['on_midget_indices'][0][0] - 1
idx_off_par = cellindex['off_parasol_indices'][0][0] - 1
idx_off_mid = cellindex['off_midget_indices'][0][0] - 1
idx_allunits = np.concatenate((idx_on_par,idx_on_mid,idx_off_par,idx_off_mid),axis=-1)
totalNum_units = idx_allunits.shape[-1]

uname_all = list()
ctr = 0
for i in idx_allunits[0,:]:
    ctr+=1
    if i in idx_on_par[0,:]:
        uname_rgb = 'on_par_%03d'%ctr
    elif i in idx_on_mid[0,:]:
        uname_rgb = 'on_mid_%03d'%ctr
    elif i in idx_off_par[0,:]:
        uname_rgb = 'off_par_%03d'%ctr
    elif i in idx_off_mid[0,:]:
        uname_rgb = 'off_mid_%03d'%ctr    
    else:
        uname_rgb = 'uncat_%03d'%ctr    
    
    uname_all.append(uname_rgb)



num_checkers_x = 39
num_checkers_y = 30
checkSize_um = -1

# units_spatRf = rgb["units_spatRf"]

# units_spatRf = units_toTake
lightLevel_text = ['scot-0.3','scot-3','scot-30']#,'scot-0.3-3','scot-0.3-30','scot-3-30']
lightLevel_mul = [0.3,3,30]
spikeTime_offset_test = {    # ms
    'monkey01': 0,
    }

spikeTime_offset = {    # ms
    'monkey01': 0,
    }

samps_shift = {    # ms
    'monkey01': 0,
    }


lightLevel_idx = 0
# %%
for lightLevel_idx in [0,1,2]:     #[0=0.3; 1=3; 2=30]
# %%
    lightLevel = lightLevel_text[lightLevel_idx]
    t_frame = 16.6 #ms
       

    # % Test set
    # This form of reshaping is to be compatible with data_handler snipper i wrote for kierstens data
    stim_frames = matData[lightLevel_idx][0]['rep_wn_movie'][0][0]
    stim_frames = np.moveaxis(stim_frames,-1,0)
    stim_frames = np.reshape(stim_frames,(stim_frames.shape[0],stim_frames.shape[1]*stim_frames.shape[2]),order='F')
    # rgb = np.reshape(stim_frames,(596,num_checkers_y,num_checkers_x),order='F')
    # plt.imshow(rgb[0,:,:])
    
    stim_frames[stim_frames>0.5] = 1
    stim_frames[stim_frames<0.5] = 0
    
    
    stim_fac = 4
    stim_frames = stim_frames[:int(stim_frames.shape[0]/stim_fac)]
    stim_frames = np.repeat(stim_frames,stim_fac,axis=0)
    



    
    if UP_SAMP == 1:
        upSampFac = int(np.floor(t_frame))
    else:
        upSampFac = int(np.floor(t_frame/bin_width))
    t_frame_orig = t_frame
    
    if upSampFac > 1:
        stim_frames = np.repeat(stim_frames,upSampFac,axis=0) # ~69 minutes of stimuli
        t_frame = t_frame/upSampFac   # now each frame is 16 ms because we upsampled

    
    if CONVERT_RSTAR == True:
        meanIntensity = lightLevel_mul[lightLevel_idx]
        stim_frames = applyLightIntensities(meanIntensity,stim_frames,t_frame)
    

    
    spikeTimes_allTrials = np.array(matData[lightLevel_idx][0]['rep_spikes'][0][0]) #matData['OFFbs_info']['all_spikes'][0][0][0][lightLevel_idx][0]*1000 # in ms
    numTrials = matData[lightLevel_idx][0]['rep_spikes'][0][0][0][0].shape[0]

    stimLength = stim_frames.shape[0]
    
    flips = np.arange(0,(stimLength+1)*t_frame,t_frame)
    
    if UP_SAMP==1:
        sig = 40
    else:
        sig = 4#8 # 2   #2 for 16 ms; 4 for 8 ms; 8 for 4 ms; 20 for 1 ms
    
    spikeRate_cells = np.empty((stimLength,totalNum_units,numTrials))
    spikeCounts_cells = np.empty((stimLength,totalNum_units,numTrials))
    spikeTimes_cells = np.empty((totalNum_units,numTrials),dtype='object')
    
    
    ctr_units = -1
    for U in idx_allunits[lightLevel_idx,:]:
        ctr_units += 1


        for tr in range(numTrials):
            startTime = 0#stimulus_start_times[tr]
            endTime = startTime + flips[-1]
            
            spikeTimes_unit = np.floor(np.squeeze(spikeTimes_allTrials[U,0][tr,0])*1000)
            rgb = np.logical_and(spikeTimes_unit>startTime,spikeTimes_unit<endTime)     # Unit Type here
            spikeTimes = spikeTimes_unit[rgb]
            spikeTimes = spikeTimes - startTime
            if UP_SAMP == 1:
                spikeRate, spikeCounts = spiketools.MEA_spikerates(spikeTimes,sig,stimLength)
            else:
                spikeRate, spikeCounts = spiketools.MEA_spikerates_binned(spikeTimes,sig,flips,t_frame)
            # plt.plot(spikeRate)
        
            spikeRate_cells[:,ctr_units,tr] = spikeRate
            spikeCounts_cells[:,ctr_units,tr] = spikeCounts
            spikeTimes_cells[ctr_units,tr] = np.array(spikeTimes)
            
            
    spikeRate_test = (spikeRate_cells,spikeCounts_cells,spikeTimes_cells)   # dims [stim_files][0=spikeRate,1=spikeCounts,2=spikeTimes]
    stim_frames_test = (stim_frames,flips)

    # %% Training set
    
    stim_frames_orig = matData[lightLevel_idx][0]['white_noise_movie'][0][0]
    stim_frames_orig = np.moveaxis(stim_frames_orig,-1,0)
    stim_frames = np.reshape(stim_frames_orig,(stim_frames_orig.shape[0],stim_frames_orig.shape[1]*stim_frames_orig.shape[2]),order='F')
    # rgb = np.reshape(stim_frames,(stim_frames.shape[0],num_checkers_y,num_checkers_x),order='F')
    # plt.imshow(rgb[0,:,:])
    
    
    training_stim_fac = 4
    stim_frames = np.repeat(stim_frames,training_stim_fac,axis=0)
    
    if CONVERT_RSTAR == True:
        meanIntensity = lightLevel_mul[lightLevel_idx]
        stim_frames = applyLightIntensities(meanIntensity,stim_frames,t_frame)

    
    trig_sigs = np.squeeze(matData[lightLevel_idx][0]['non_rep_triggers'][0][0]) #seconds
    spikeTimes_mat = np.squeeze(matData[lightLevel_idx][0]['non_rep_spikes'][0][0])
    # rgb = np.diff(trig_sigs)
    # idx_long = np.where(rgb>2)[0]
    
   
    if upSampFac > 1:
        stim_frames = np.repeat(stim_frames,upSampFac,axis=0) # ~69 minutes of stimuli

    
    stimLength = stim_frames.shape[0]
    flips = np.arange(0,(stimLength+1)*t_frame,t_frame)
    stimLength_orig = stim_frames_orig.shape[0]*training_stim_fac
    spikeTimes_vec = np.arange(0,(stimLength_orig+1)*t_frame_orig,t_frame_orig)

    
    spikeRate_cells = np.empty((stimLength,totalNum_units))
    spikeCounts_cells = np.empty((stimLength,totalNum_units))
    spikeTimes_cells = np.empty(totalNum_units,dtype='object')
    
    idx_start = 10000#118
    startTime = np.floor(trig_sigs[0]*1000)
    

    
    # cells
    ctr_units = -1
    for U in idx_allunits[lightLevel_idx,:]:
        ctr_units += 1
        spikeTimes = np.where(np.squeeze(spikeTimes_mat[U]>0))[0]
        spikeTimes = spikeTimes_vec[spikeTimes]
        # spikeTimes = spikeTimes - startTime
        spikeTimes = spikeTimes - spikeTime_offset[expDate]     # somethings wrong with alignment of retina3 and has to be manually fixed here
        spikeTimes = spikeTimes[spikeTimes>flips[0]]
        if UP_SAMP == 1:
            spikeRate, spikeCounts = spiketools.MEA_spikerates(spikeTimes,sig,stimLength)
        else:
            spikeRate, spikeCounts = spiketools.MEA_spikerates_binned(spikeTimes,sig,flips,t_frame)

        # plt.plot(spikeRate)
        idx_plot = np.arange(5000,18000)
        plt.plot(spikeRate[idx_plot])
        rgb = spikeCounts.copy().astype('int32')
        # rgb[rgb<1] = np.nan
        plt.plot((rgb[idx_plot]*spikeRate[idx_plot].max()+1),'ro')

    
        spikeRate_cells[:,ctr_units] = spikeRate    # adjust for discarding initial data in the end of this cell
        spikeCounts_cells[:,ctr_units] = spikeCounts
        spikeTimes_cells[ctr_units] = np.array(spikeTimes[spikeTimes>flips[idx_start]])
        
    
    
    
    spikeRate_cells = spikeRate_cells[idx_start:,:]
    spikeCounts_cells = spikeCounts_cells[idx_start:,:]
    # spikeTimes_cells = spikeTimes_cells[spikeTimes_cells>flips[idx_start]]
    spikeRate_train = (spikeRate_cells,spikeCounts_cells,spikeTimes_cells)   # dims [stim_files][0=spikeRate,1=spikeCounts,2=spikeTimes]
    
    stim_frames = stim_frames[idx_start:,:]
    flips = flips[idx_start:]
    stim_frames_train = (stim_frames,flips)
    
    
    a = np.unique(stim_frames_train[0],axis=0)
    b = np.unique(stim_frames_test[0],axis=0)
    c = np.concatenate((b,a),axis=0)
    d = np.unique(c,axis=0)
    if d.shape[0] != c.shape[0]:
        raise ValueError('training samples contains validation samples') 
    
    # %% Save dataset
    f = h5py.File(fname_save, 'a')
    try:
        f.create_dataset('expDate',data=np.array(expDate,dtype='bytes'))
        f.create_dataset('units',data=np.array(uname_all,dtype='bytes'))
    except:
        pass
    # f.create_dataset('same_stims',data=np.array(same_stims))
    
    # Training dataset
    # f.create_dataset('/'+lightLevel+'/units',data=np.array(uname_all,dtype='bytes'))

    grp = f.create_group('/'+lightLevel+'/train')
    d = grp.create_dataset('stim_frames',data=stim_frames_train[0],compression='gzip')
    d.attrs['num_checkers_x'] = num_checkers_x
    d.attrs['num_checkers_y'] = num_checkers_y
    d.attrs['checkSize_um'] = checkSize_um
    d.attrs['t_frame'] = t_frame
    
    
    d = grp.create_dataset('flips_timestamp',data=stim_frames_train[1],compression='gzip')
    d.attrs['time_unit'] = 'ms' 
    
    d = grp.create_dataset('spikeRate',data=spikeRate_train[0],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['sig'] = sig
    
    d = grp.create_dataset('spikeCounts',data=spikeRate_train[1],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    
    # Test dataset
    grp = f.create_group('/'+lightLevel+'/val')
    d = grp.create_dataset('stim_frames',data=stim_frames_test[0],compression='gzip')
    d.attrs['num_checkers_x'] = num_checkers_x
    d.attrs['num_checkers_y'] = num_checkers_y
    d.attrs['checkSize_um'] = checkSize_um
    d.attrs['t_frame'] = t_frame
    
    d = grp.create_dataset('flips_timestamp',data=stim_frames_test[1],compression='gzip')
    d.attrs['time_unit'] = 'ms' 
    
    d = grp.create_dataset('spikeRate',data=spikeRate_test[0],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['sig'] = sig
    d.attrs['num_trials'] = numTrials
    
    d = grp.create_dataset('spikeCounts',data=spikeRate_test[1],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    

    f.close()
        
                           
    
# %% Combine lightlevels
f = h5py.File(fname_save, 'a')

lightLevels_comb_pairs = np.array([[0,1],[0,2],[1,2]])


num_units = len(uname_all)
num_val_trials = spikeRate_test[0].shape[-1]

for c in range(0,lightLevels_comb_pairs.shape[0]):
    stim_frames_train = np.array([],dtype='float32').reshape(0,num_checkers_y*num_checkers_x)
    flips_timestamp_train = np.array([],dtype='float32')
    spikeRate_train = np.array([],dtype='float32').reshape(0,num_units)
    spikeCounts_train = np.array([],dtype='float32').reshape(0,num_units)
    
    stim_frames_val = np.array([],dtype='float32').reshape(0,num_checkers_y*num_checkers_x)
    flips_timestamp_val = np.array([],dtype='float32')
    spikeRate_val = np.array([],dtype='float32').reshape(0,num_units,num_val_trials)
    spikeCounts_val = np.array([],dtype='float32').reshape(0,num_units,num_val_trials)

    
    rgb = lightLevels_comb_pairs[c]
    lightLevel = 'scot'
    for i in rgb:
        lightLevel = lightLevel+lightLevel_text[i][4:]
    
    
    for k in lightLevels_comb_pairs[c,:]:
        # train data
        rgb = np.array(f[lightLevel_text[k]+'/train/stim_frames'])
        stim_frames_train = np.concatenate((stim_frames_train,rgb),0)
        
        rgb = np.array(f[lightLevel_text[k]+'/train/flips_timestamp'])
        flips_timestamp_train = np.concatenate((flips_timestamp_train,rgb),0)
    
        rgb = np.array(f[lightLevel_text[k]+'/train/spikeRate'])
        assert(rgb.shape[1]==spikeRate_train.shape[1],'num units dont match')
        spikeRate_train = np.concatenate((spikeRate_train,rgb),0)
        
        rgb = np.array(f[lightLevel_text[k]+'/train/spikeCounts'])
        spikeCounts_train = np.concatenate((spikeCounts_train,rgb),0)
       
        # val data
        rgb = np.array(f[lightLevel_text[k]+'/val/stim_frames'])
        stim_frames_val = np.concatenate((stim_frames_val,rgb),0)
        
        rgb = np.array(f[lightLevel_text[k]+'/val/flips_timestamp'])
        flips_timestamp_val = np.concatenate((flips_timestamp_val,rgb),0)
    
        rgb = np.array(f[lightLevel_text[k]+'/val/spikeRate'])
        assert(rgb.shape[1]==spikeRate_val.shape[1],'num units dont match')
        spikeRate_val = np.concatenate((spikeRate_val,rgb),0)
        
        rgb = np.array(f[lightLevel_text[k]+'/val/spikeCounts'])
        spikeCounts_val = np.concatenate((spikeCounts_val,rgb),0)

        # f.copy(lightLevel_text[k]+'/val/',grp_data)
        # f.move(lightLevel+'/val',lightLevel+'/val_'+lightLevel_text[k])
        
        f[lightLevel_text[k]+'/val/spikeRate'].attrs['samps_shift'] = samps_shift[expDate]/bin_width
        
    
    grp = f.create_group(lightLevel+'/train')
    d = grp.create_dataset('stim_frames',data=stim_frames_train,compression='gzip')
    d.attrs['num_checkers_x'] = num_checkers_x
    d.attrs['num_checkers_y'] = num_checkers_y
    d.attrs['checkSize_um'] = checkSize_um
    d.attrs['t_frame'] = t_frame
    
    
    d = grp.create_dataset('flips_timestamp',data=flips_timestamp_train,compression='gzip')
    d.attrs['time_unit'] = 'ms' 
    
    d = grp.create_dataset('spikeRate',data=spikeRate_train,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['sig'] = sig
    
    d = grp.create_dataset('spikeCounts',data=spikeCounts_train,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    
    
    
    # Val
    grp = f.create_group(lightLevel+'/val')
    d = grp.create_dataset('stim_frames',data=stim_frames_val,compression='gzip')
    d.attrs['num_checkers_x'] = num_checkers_x
    d.attrs['num_checkers_y'] = num_checkers_y
    d.attrs['checkSize_um'] = checkSize_um
    d.attrs['t_frame'] = t_frame
    
    d = grp.create_dataset('flips_timestamp',data=flips_timestamp_val,compression='gzip')
    d.attrs['time_unit'] = 'ms' 
    
    d = grp.create_dataset('spikeRate',data=spikeRate_val,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['sig'] = sig
    d.attrs['num_trials'] = numTrials
    
    d = grp.create_dataset('spikeCounts',data=spikeCounts_val,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)


f.close()
   
    
    
              
# %% plot STAs to check data
from pyret.filtertools import sta, decompose
idx_cell = 0
# dataset = 'train'
stim = stim_frames_train[0]
spikes = spikeRate_train[2][idx_cell]

stim = np.reshape(stim,(stim.shape[0],num_checkers_y,num_checkers_x),order='F')       
flips = stim_frames_train[1]


sta_cell,_ = sta(flips, stim, spikes, 60)

spatial_feature, temporal_feature = decompose(sta_cell)
plt.imshow(spatial_feature,cmap='winter');plt.show()
plt.plot(temporal_feature);plt.show()


# idx_trial = 224
# stim = stim_frames_test[0]
# spikes = spikeRate_test[2][idx_cell,idx_trial]

# stim = np.reshape(stim,(stim.shape[0],num_checkers_y,num_checkers_x),order='F')       
# flips = stim_frames_test[1]
# spikeRate = spikeRate_test[0][:,idx_cell,idx_trial]

# spikeCounts = spikeRate_test[1][:,idx_cell,idx_trial]
# spikeCounts = spikeCounts>0
# rgb = flips[:-1]
# spikeTimes_binned = rgb[spikeCounts]


# sta_cell,_ = sta(flips, stim, spikes, 40)

# spatial_feature, temporal_feature = decompose(sta_cell)
# # plt.imshow(spatial_feature,cmap='winter')
# plt.plot(temporal_feature)

# plt.show()

# %% RWA
from pyret.filtertools import sta, decompose
from model.data_handler import rolling_window

idx_cell = 5
t_start = 0
t_end = 20000
temporal_window = 60
# dataset = 'train'
stim = stim_frames_train[0][t_start:t_end]

stim = np.reshape(stim,(stim.shape[0],num_checkers_y,num_checkers_x),order='F')       
flips = stim_frames_train[1][t_start:t_end]
spikeRate = spikeRate_train[0][t_start:t_end,idx_cell]

stim = rolling_window(stim,temporal_window)
spikeRate = spikeRate[temporal_window:]
rwa = np.nanmean(stim*spikeRate[:,None,None,None],axis=0)



spatial_feature, temporal_feature = decompose(rwa)
# plt.imshow(spatial_feature,cmap='winter')
plt.plot(temporal_feature)


# %%
# plt.plot(scot_temp)
# plt.plot(temporal_feature)
# plt.show()