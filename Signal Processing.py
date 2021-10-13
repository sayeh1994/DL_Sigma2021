# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:28:56 2021

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
from sklearn.metrics import mean_squared_error
#%% Load Data
path = r"E:\Subjects\Master 02\Semester 1\Deep Learning\Code Project\Audio Path\\"
CleanAduio="247906.mp3"
NoiseAudio = "Washing Machine Spining Fast - QuickSounds.com.mp3"
clean_data, rate = librosa.load(path+CleanAduio)
noise_data, rate_n,  = librosa.load(path+NoiseAudio)
#%% Checking SFTF
s_stft = stft.stft(clean_data,128)
s_hat = stft.istft(s_stft,128)
MSE = mean_squared_error(clean_data[:len(s_hat)], s_hat)
print("the error between inverse sftf of the original signal with itself is = ",MSE)
#%% Extract a noise segment from a random location in the noise file
ind = np.random.randint(len(noise_data)-len(clean_data)+1)
noiseSegment = noise_data[ind:ind+len(clean_data)]
#%%
speechPower = sum(clean_data**2)
noisePower = sum(noiseSegment**2)
noisyAudio = clean_data + np.sqrt(speechPower/noisePower) * noiseSegment
#%%
plt.figure()
plt.plot(clean_data),plt.title("Clean Audio")
plt.figure()
plt.plot(noisyAudio),plt.title("Noisy Audio")
#%% writing noisy audio in a wav file
path_result = r"E:\Subjects\Master 02\Semester 1\Deep Learning\Code Project\Result\\"
wavfile.write(path_result+"Noise.wav", rate, noisyAudio)
#%%
