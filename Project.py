# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 08:21:06 2021

@author: Sayeh
"""

import noisereduce as nr
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
#from librosa.core import stft, istft
import stft
from scipy import fft
from scipy import signal as sig
import soundfile as sf
import librosa
#%%
def minmax(s):
    s = s/max(s)
    return s
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
#%% load data
path = r"E:\Subjects\Master 02\Semester 1\Deep Learning\Code Project\Audio Path"
data, rate = librosa.load(path+"\\247906.mp3")
data_n, rate_n,  = librosa.load(path+"\\Washing Machine Spining Fast - QuickSounds.com.mp3")
#%%
f, t, Zxx = sig.stft(data, rate, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin = np.min(np.abs(Zxx)), vmax = np.max(np.abs(Zxx)), shading='green')
plt.title("Clean Audio")
f, t, Zxx = sig.stft(data_n[:,0], rate, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin = np.min(np.abs(Zxx)), vmax = np.max(np.abs(Zxx)), shading='green')
plt.title("Noise")
#%%
plt.figure()
plt.plot(data),plt.title("Clean")
plt.figure()
plt.plot(np.abs(fft(data_n[:,0]))),plt.title("Noise")
#%%
N = len(data)
N_n = len(data_n[:,0])

data_norm = minmax(data)
#%% Visualizing the signal
plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data)
plt.title("Clear Signal")
#%%
plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data_n[:,0])
plt.title("Noise Signal")


#%% normalizing noise to the data range
noise = []
wgn = True
if wgn is False:    
    if N>N_n:
        count = int(N/N_n)+1
        for c in count: 
            for i in range(N_n):
                noise.append(data_n[i,0])
    else: 
        for i in range(N):
            noise.append(data_n[i,0])
    noise = minmax(noise)
else:
    noise = np.random.random(size=N)

plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(noise)

#%% adding noise

snr = signaltonoise(data)
print(snr)
#%%
alpha = 0.1
noisy_audio = data_norm + alpha*noise

plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(noisy_audio)

snr = signaltonoise(noisy_audio)
print(snr)
#%% Denoising
# perform noise reduction
reduced_noise = nr.reduce_noise(y=noisy_audio, sr=rate)

plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(reduced_noise)

snr = signaltonoise(reduced_noise)
print(snr)
#%%
path_result = r"E:\Subjects\Master 02\Semester 1\Deep Learning\Code Project\Result\\"
wavfile.write(path_result+"Noise.wav", rate, noisy_audio)
wavfile.write(path_result+"DeNoise.wav", rate, reduced_noise)
#%%
spectrum = stft.stft(data,128)
back_y = stft.istft(spectrum,128)

plt.figure()
plt.plot(abs(noisy_audio),marker=".")

#%%
spectrum2 = fft(data)
spectrum3 = sig.stft(data)
plt.figure()
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(abs(spectrum3[2]))
#%%
f, t, Zxx = sig.stft(data, rate, nperseg=1000)
#plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.pcolormesh(t, f, np.abs(Zxx), vmin = np.min(np.abs(Zxx)), vmax = np.max(np.abs(Zxx)), shading='green')
