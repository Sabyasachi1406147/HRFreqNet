# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:50:23 2023

@author: sb3682
"""

#generating training dataset
import numpy as np
from data import fr_v2
from data import fr
import matplotlib.pyplot as plt
from random import shuffle
import h5py
import scipy.signal

fr_size = 1000 # size of frequency representation
max_n_freq = 15 # for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq
signal_dim = 200 # dimensionof the input signal
min_sep = 1 # minimum separation between spikes, normalized by signal_dim
distance = 'normal' # distance distribution between spikes
amplitude = 'normal_floor' # spike amplitude distribution
floor_amplitude = 0.8 # minimum amplitude of spikes
gaussian_std = 0.1 # std of the gaussian kernel normalized by signal_dim
kernel_type = 'gaussian_v2' # type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]

triangle_slope = 4000 # slope of the triangle kernel normalized by signal_dim
n_training = 600 # number of training data
n_validation = 60 # number of validation data
batch_size = 256 # batch size used during training
o = 20
snr = [o , o, o, o, o] #snr in DB
# snr = np.repeat(snr, int(n_training/10000))
kernel_param = gaussian_std / signal_dim

def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (1 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        a = (2-0.3)*np.abs(np.random.rand(*dim)) + floor_amplitude
        return a #np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)

def frequency_generator(f, nf, min_sep, dist_distribution):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05): # f = initial value of freq. = 0
                                             # nf = nfreq(sample_number)
                                             # min_sep = 1 (normalized by signal dimension)
                                             # min_freq_separation = min_sep / signal_dimension
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5 # This gives you a value between -0.5 to 0.5. 
                                     # the unform() function generates value between 0 to 1
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale) # normal() randomly distributes numbers in gaussian
                                              # scale is the standard deviation
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5 # np.sign gives the sign of the variable
                                                                          # %1 = [0.23 % 1 = 0.23 ; -0.23 % 1 = 0.77; 
                                                                          #       0.89 % 1 = 0.11; -0.89 % 1 = 0.11]
                                                                          # so -0.5 <= f_new <= 0.5  
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new

def gen_signal_with_noise(num_samples, signal_dim, num_freq, min_sep, snr, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    
    # signal_dim = 50
    # num_freq = max_num_frequencies = 10
    # min_sep = 1
    assigned_snr = np.array([])
    calculated_snr = np.array([])
    s_complex = np.zeros((1, signal_dim))
    # s = np.zeros((num_samples, 2, signal_dim)) # 2 represents real and imag
    # n = np.zeros((num_samples, 2, signal_dim)) # 2 represents real and imag
    xgrid = np.arange(signal_dim)[:, None] # grid_size (50,)
    f = np.ones((num_samples, num_freq)) * np.inf # 2D array of infinite values
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude) #generating amplitude
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi # in radians
    d_sep = min_sep / signal_dim # minimum separation between two frequency
    nfreq = []
    amp = np.zeros((num_samples, num_freq))
    amp_norm = np.zeros((num_samples, num_freq))
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
        #print(nfreq)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        frequency_generator(f[n], nfreq[n], d_sep, distance)
        s_complex = np.zeros((1,signal_dim))
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * f[n, i] * xgrid.T)
            amp[n,i] = r[n,i]
            
            # signal = A*e(j(theta)+ 2*pi*f*t)
            s_complex = s_complex + sin
        amp_norm[n] = amp[n]/np.max(amp[n])
        s_complex = s_complex / np.sqrt(np.mean((np.abs(s_complex))**2))
        #print(s_complex.shape)
        s_power = np.mean((np.abs(s_complex))**2)
        # generate gaussian noise
        n_complex = np.random.randn(signal_dim, 2).view(np.complex128)
        n_complex = n_complex / np.sqrt((np.mean((np.abs(n_complex))**2))*(10**(snr/10)))
        n_power = np.mean((np.abs(n_complex))**2)
        noisy_signal = s_complex + n_complex.T
        
        snr_calc = 10*np.log10(np.mean((np.abs(s_complex))**2)/ np.mean((np.abs(noisy_signal-s_complex))**2))
        assigned = float(snr)
        calculated = float(snr_calc)
        assigned_snr= np.append(assigned_snr,assigned)
        calculated_snr = np.append(calculated_snr,calculated)
        if n == 0:
            noise = noisy_signal[None]
            s = s_complex[None]
            
        else:
            noise = np.concatenate((noise, noisy_signal[None]),axis=0)
            s = np.concatenate((s, s_complex[None]),axis=0)
    noisy_signal = np.concatenate((noise.real,noise.imag),axis=1)
    #print(noisy_signal.shape)
    
    clean_signal = np.concatenate((s.real,s.imag),axis=1)
    
    # f.sort(axis=1)
    f[f == float('inf')] = -10
    return noisy_signal.astype('float32'), clean_signal.astype('float32'), f.astype('float32'), nfreq, assigned_snr, calculated_snr, amp.astype('float32'), amp_norm.astype('float32')

def load_dataloader_fixed_noise(num_samples, signal_dim, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                                kernel_type, kernel_param, batch_size, xgrid, snr, noise):
    noisy_signals, clean_signals, f, nfreq, assigned_snr, calculated_snr, amp, amp_norm = gen_signal_with_noise(num_samples, signal_dim, max_n_freq, min_sep, snr, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True)
    frequency_representation = fr_v2.freq2fr(f, xgrid, amp, kernel_type, kernel_param)
    frequency_representation_2 =  fr.freq2fr(f, xgrid, kernel_type, kernel_param)
    frequency_representation_cres = fr_v2.freq2fr(f, xgrid, amp_norm, kernel_type, kernel_param)
    
    #noisy_signals = np.concatenate((noisy_signals.real, noisy_signals.imag),axis=1)
    
    return noisy_signals, clean_signals, f, nfreq, frequency_representation, frequency_representation_2, frequency_representation_cres, assigned_snr, calculated_snr, amp, amp_norm

xgrid_fr = np.linspace(-0.5, 0.5, fr_size, endpoint=True)
xgrid_td = np.linspace(0, signal_dim, num=signal_dim)
if kernel_type == 'triangle':
    kernel_param = triangle_slope / signal_dim
else:
    kernel_param = gaussian_std / signal_dim

for i in range(len(snr)):
    n_training_snr = int(n_training/len(snr))
    snr_idx = snr[i]
    print(snr_idx)
    noisy_signals_training, clean_signals_training, f_train, nfreq_training, frequency_representation_training, frequency_representation_2_training, frequency_representation_cres_training, assigned_snr, calculated_snr, amp, amp_norm = load_dataloader_fixed_noise(n_training_snr, 
                            signal_dim=signal_dim, max_n_freq=max_n_freq,
                            min_sep=min_sep, distance=distance, amplitude=amplitude,
                            floor_amplitude=floor_amplitude, kernel_type=kernel_type,
                            kernel_param=kernel_param, batch_size=batch_size, xgrid=xgrid_fr, snr = snr_idx, noise = 'gaussian_blind')

    if i == 0:
        noisy_signal_train = noisy_signals_training
        clean_signal_train = clean_signals_training
        freq_train = f_train
        fr_train = frequency_representation_training
        snr_train_assigned = assigned_snr[None]
        snr_train_calculated = calculated_snr[None]
        nfreq_train = nfreq_training
    else:
        noisy_signal_train = np.concatenate((noisy_signal_train, noisy_signals_training),axis = 0)
        clean_signal_train = np.concatenate((clean_signal_train, clean_signals_training),axis = 0)
        freq_train = np.concatenate((freq_train, f_train),axis = 0)
        fr_train = np.concatenate((fr_train, frequency_representation_training),axis = 0)
        snr_train_assigned = np.concatenate((snr_train_assigned, assigned_snr[None]),axis = 1)
        snr_train_calculated = np.concatenate((snr_train_calculated, calculated_snr[None]),axis = 1)
        snr_differences = np.concatenate((snr_train_assigned, snr_train_calculated),axis = 0)
        snr_differences = snr_differences.T
        nfreq_train = np.concatenate((nfreq_training, nfreq_train),axis = 0)
        
        
for i in range(len(snr)):
    n_validation_snr = int(n_validation/len(snr))
    snr_idx = snr[i]
    noisy_signals_validation, clean_signals_validation, f_validation, nfreq_validation, frequency_representation_validation, frequency_representation_2_validation, frequency_representation_cres_validation, assigned_snr, calculated_snr, amp, amp_norm = load_dataloader_fixed_noise(n_validation_snr, 
                            signal_dim=signal_dim, max_n_freq=max_n_freq,
                            min_sep=min_sep, distance=distance, amplitude=amplitude,
                            floor_amplitude=floor_amplitude, kernel_type=kernel_type,
                            kernel_param=kernel_param, batch_size=batch_size, xgrid=xgrid_fr, snr = snr_idx, noise = 'gaussian_blind')
    if i == 0:
        noisy_signal_val = noisy_signals_validation
        clean_signal_val = clean_signals_validation
        freq_val = f_validation
        fr_val = frequency_representation_validation
        fr_val_2 = frequency_representation_2_validation
        fr_val_cres = frequency_representation_cres_validation
        snr_val_calculated = calculated_snr
        nfreq_val = nfreq_validation
    else:
        noisy_signal_val = np.concatenate((noisy_signal_val, noisy_signals_validation),axis = 0)
        clean_signal_val = np.concatenate((clean_signal_val, clean_signals_validation),axis = 0)
        freq_val = np.concatenate((freq_val, f_validation),axis = 0)
        fr_val = np.concatenate((fr_val, frequency_representation_validation),axis = 0)
        fr_val_2 = np.concatenate((fr_val_2,frequency_representation_2_validation),axis = 0)
        fr_val_cres = np.concatenate((fr_val_cres,frequency_representation_cres_validation),axis = 0)
        snr_val_calculated = np.concatenate((snr_val_calculated, calculated_snr),axis = 0)
        nfreq_val = np.concatenate((nfreq_val, nfreq_validation),axis = 0)

snr_train = snr_train_calculated.T
snr_val = snr_val_calculated.T

# shuffle_idx_train = list(range(n_training))
# shuffle_idx_val = list(range(n_validation))
# shuffle(shuffle_idx_train)
# shuffle(shuffle_idx_val)

# noisy_signal_train = noisy_signal_train[shuffle_idx_train]
# clean_signal_train = clean_signal_train[shuffle_idx_train]
# freq_train = freq_train[shuffle_idx_train]
# fr_train = fr_train[shuffle_idx_train]
# snr_train = snr_train[shuffle_idx_train]

# noisy_signal_val = noisy_signal_val[shuffle_idx_val]
# clean_signal_val = clean_signal_val[shuffle_idx_val]
# freq_val = freq_val[shuffle_idx_val]
# fr_val = fr_val[shuffle_idx_val]
# snr_val = snr_val[shuffle_idx_val]

idx = 0
plt.figure(1)
plt.subplot(311)
plt.plot(xgrid_td,clean_signal_train[idx,0, :],color='b')
plt.xlabel("Real Signal")
plt.subplot(312)
plt.plot(xgrid_td,clean_signal_train[idx,1, :],color= 'r')
plt.xlabel("Imag Signal")

clean_complex = clean_signal_val[idx,0, :] + clean_signal_val[idx,1, :]*1j
clean_fft = np.fft.fftshift(np.fft.fft(clean_complex,fr_size))
plt.subplot(313)
plt.plot(xgrid_fr,abs(clean_fft),color= 'c')
plt.xlabel("FFT")

plt.figure(2)
plt.subplot(311)
plt.plot(xgrid_td,noisy_signal_val[idx,0, :],color='b')
plt.xlabel("Real Signal")
plt.subplot(312)
plt.plot(xgrid_td,noisy_signal_val[idx,1, :],color= 'r')
plt.xlabel("Imag Signal")

noise_complex = noisy_signal_val[idx,0, :] + noisy_signal_val[idx,1, :]*1j
noise_fft = np.fft.fftshift(np.fft.fft(noise_complex,fr_size))
plt.subplot(313)
plt.plot(xgrid_fr,abs(noise_fft),color= 'c')
plt.xlabel("FFT")

fr_p = fr_val[idx,:]

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

fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/checkpoint/experiment_name/fr/final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util.load(fr_path, 'fr', device)
fr_module.cpu()
fr_module.eval()

noise = noisy_signal_val[idx]
with torch.no_grad():
    RD_clean, RD_fr = fr_module(torch.tensor(noise[None]))

plt.figure(5)
plt.subplot(311)
plt.plot(xgrid_td,RD_clean[0][0][:200],color='b')
plt.xlabel("Real Signal")
plt.subplot(312)
plt.plot(xgrid_td,RD_clean[0][0][200:],color= 'r')
plt.xlabel("Imag Signal")
clean = RD_clean[0][0][:200] + RD_clean[0][0][200:]*1j
clean_fft = np.fft.fftshift(np.fft.fft(clean,fr_size))
plt.subplot(313)
plt.plot(xgrid_fr,abs(clean_fft),color= 'c')
plt.xlabel("FFT")


plt.figure(6)
plt.subplot(2,3,1)
plt.plot(xgrid_fr,3*fr_val[idx,:],color='b')
# plt.plot(xgrid_fr,fr_val_2[idx,:],color='k')
plt.xlabel("fr_representation_UNET")

plt.subplot(2,3,2)
plt.plot(xgrid_fr,fr_val_2[idx,:],color='b')
# plt.plot(xgrid_fr,fr_val_2[idx,:],color='k')
plt.xlabel("fr_representation_deep")

plt.subplot(2,3,3)
plt.plot(xgrid_fr,fr_val_cres[idx,:],color='b')
# plt.plot(xgrid_fr,fr_val_2[idx,:],color='k')
plt.xlabel("fr_representation_cvDeep")

RD_fr = RD_fr.numpy()
RD_fr[RD_fr < 0.05] = 0
plt.subplot(2,3,4)
plt.plot(xgrid_fr,abs(RD_fr[0]),color= 'c')
plt.xlabel("FFT")

fr_path_2 = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/DeepFreq-main/DeepFreq-master/checkpoint/experiment_name/fr/final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util_deep.load(fr_path_2, 'fr', device)
fr_module.cpu()
fr_module.eval()

noise = noisy_signal_val[idx]
with torch.no_grad():
    RD_fr = fr_module(torch.tensor(noise[None]))
    
RD_fr = RD_fr.numpy()
plt.subplot(2,3,5)
plt.plot(xgrid_fr,abs(RD_fr[0]),color= 'c')
plt.xlabel("FFT")

fr_path_3 = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/cResFreq-main/cResFreq-main/Python codes/RDN-1D/checkpoint/layer1_big8/fr/final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util_cfreq.load(fr_path_3, 'fr', device)
fr_module.cpu()
fr_module.eval()

noise = noisy_signal_val[idx]
with torch.no_grad():
    RD_fr = fr_module(torch.tensor(noise[None]))
    
RD_fr = RD_fr.numpy()
plt.subplot(2,3,6)
plt.plot(xgrid_fr,abs(RD_fr[0]),color= 'c')
plt.xlabel("FFT")