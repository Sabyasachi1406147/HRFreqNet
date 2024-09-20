# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:53:56 2023

@author: sb3682
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
import numpy.matlib
import scipy.io as sio
import torch
import util
import util_deep
import util_cfreq
import scipy.signal as signal
import h5py
import time

# mat = sio.loadmat("D:/110 ASL Data/Raw1D/class_15_Alperen_3.mat")
# RPExt_2 = mat["data1D_cut"]

data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TFA-Net_train/generate_data/train_data.h5", 'r')
x = np.asarray(data['signal_train'])
x = x[25]
RPExt_2 = x[0] + 1j*x[1]
RPExt_2 = RPExt_2[None]
#Using fr_module
fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/checkpoint/experiment_name/fr/epoch_70.pth'
# fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/DeepFreq-main/DeepFreq-master/checkpoint/experiment_name/fr/epoch_10.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util.load(fr_path, 'fr', device)
fr_module.cpu()
fr_module.eval()

# fpass = 2000

# # Design the highpass filter
# b, a = signal.butter(4, fpass, 'high', fs=7000000)

# for i in range(len(RPExt_2)):
#     RPExt_2[i, :] = signal.lfilter(b, a, RPExt_2[i, :])
    
rd_window_size = 120
hop_size = 20
low_lim = -160
num_frames = len(RPExt_2[0])
loop = 1 + int((num_frames - rd_window_size) / hop_size)
spec_n = np.zeros((loop, 4096), dtype=np.complex64)
# spectrogram[n] = spec_n.T.astype(np.float32)
start_time = time.time()
for i in range(loop):
    start = i * hop_size
    end = start + rd_window_size
    frame = RPExt_2[0][start:end] * np.hanning(rd_window_size)
    spec_n[i] = np.fft.fftshift(np.fft.fft(frame,4096))
end_time = time.time()
time_spect = end_time-start_time
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(20*np.log((abs(spec_n.T)/np.max(abs(spec_n)))), cmap='jet',aspect='auto')
# plt.ylim(40, 160)
plt.colorbar()
plt.clim(0,low_lim)

rd_window_size = 200
hop_size = 20
loop = 1 + int((num_frames - rd_window_size) / hop_size)
frame_module = np.zeros((2,rd_window_size))
MD_fr = np.zeros((loop, 1000))
start_time = time.time()
for i in range(loop):
    start = i * hop_size
    end = start + rd_window_size
    frame = RPExt_2[0][start:end] * np.hanning(rd_window_size)
    frame_module[0] = frame.real
    frame_module[1] = frame.imag
    frame_module = (frame_module/np.sqrt(np.mean(np.power(frame_module, 2)))).astype(np.float32())
    with torch.no_grad():
        RD_clean, RD_fr = fr_module(torch.tensor(frame_module[None]))
    MD_fr[i] = RD_fr.numpy()   
end_time = time.time()
time_unet = end_time-start_time
MD_fr[MD_fr < 0.1] = 0.0001
plt.subplot(2,2,2)
plt.imshow(20*np.log(MD_fr.T/np.max(abs(MD_fr.T))), cmap='jet',aspect='auto')
# plt.ylim(200, 800)
#plt.ylim(-1500,2500)
plt.colorbar()
plt.clim(0,low_lim)

fr_path_2 = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/DeepFreq-main/DeepFreq-master/checkpoint/experiment_name/fr/epoch_50.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util_deep.load(fr_path_2, 'fr', device)
fr_module.cpu()
fr_module.eval()

rd_window_size = 200
hop_size = 20
loop = 1 + int((num_frames - rd_window_size) / hop_size)
frame_module = np.zeros((2,rd_window_size))
MD_fr = np.zeros((loop, 1000))
start_time = time.time()
for i in range(loop):
    start = i * hop_size
    end = start + rd_window_size
    frame = RPExt_2[0][start:end] * np.hanning(rd_window_size)
    frame_module[0] = frame.real
    frame_module[1] = frame.imag
    frame_module = (frame_module/np.sqrt(np.mean(np.power(frame_module, 2)))).astype(np.float32())
    with torch.no_grad():
        RD_fr = fr_module(torch.tensor(frame_module[None]))
    MD_fr[i] = RD_fr.numpy()   
end_time = time.time()
time_deep = end_time-start_time
MD_fr[MD_fr < 0.1] = 0.0001
plt.subplot(2,2,3)
plt.imshow(20*np.log(MD_fr.T/np.max(abs(MD_fr.T))), cmap='jet',aspect='auto')
# plt.ylim(200, 800)
#plt.ylim(-1500,2500)
plt.colorbar()
plt.clim(0,low_lim)

fr_path_3 = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/cResFreq-main/cResFreq-main/Python codes/RDN-1D/checkpoint/layer1_big8/fr/epoch_20.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util_cfreq.load(fr_path_3, 'fr', device)
fr_module.cpu()
fr_module.eval()

rd_window_size = 200
hop_size = 20
loop = 1 + int((num_frames - rd_window_size) / hop_size)
frame_module = np.zeros((2,rd_window_size))
MD_fr = np.zeros((loop, 1000))
start_time = time.time()
for i in range(loop):
    start = i * hop_size
    end = start + rd_window_size
    frame = RPExt_2[0][start:end] * np.hanning(rd_window_size)
    frame_module[0] = frame.real
    frame_module[1] = frame.imag
    frame_module = (frame_module/np.sqrt(np.mean(np.power(frame_module, 2)))).astype(np.float32())
    with torch.no_grad():
        RD_fr = fr_module(torch.tensor(frame_module[None]))
    MD_fr[i] = RD_fr.numpy()   
end_time = time.time()
time_cres = end_time-start_time
MD_fr[MD_fr < 0.1] = 0.0001
plt.subplot(2,2,4)
plt.imshow(20*np.log(MD_fr.T/np.max(abs(MD_fr.T))), cmap='jet',aspect='auto')
# plt.ylim(200, 800)
#plt.ylim(-1500,2500)
plt.colorbar()
plt.clim(0,low_lim)