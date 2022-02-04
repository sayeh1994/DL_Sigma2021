import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

import IPython.display as ipd
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import zipfile


windowLength = 255
overlap      = round(0.5 * windowLength) # overlap of 50%
ffTLength    = windowLength
inputFs      = 16e3
fs           = 16e3
numFeatures  = ffTLength//2 + 1
numSegments  = 16

def normalization(inputdata,NormalizationType = StandardScaler()):
  InputShape = inputdata.shape
  if len(InputShape)>2:
    m,n,h = np.size(inputdata,0),np.size(inputdata,1), np.size(inputdata,2)
    inputdata = inputdata.reshape((m,n*h))
    scaler = NormalizationType
    scaler.fit(inputdata)
    inputdata = scaler.transform(inputdata)
    inputdata = inputdata.reshape((m,n,h))
  else:
    scaler = NormalizationType
    scaler.fit(inputdata)
    inputdata = scaler.transform(inputdata)
  return inputdata
  
  
def plot_losses(results):
    plt.plot(results.history['loss'], 'bo', label='Training loss')
    plt.plot(results.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss',fontsize=14)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.legend()
    

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(1. / mse)

def snr_avg(set1, set2):
  set1 = set1[:,:,:,0].copy()
  set2 = set2[:,:,:,0].copy()
  N1 = np.size(set1,0)
  N2 = np.size(set2,0)
  psnr_avg = []
  if N1 != N2:
    return "The size of two subsets are not match"
  else:
    for i in range(N1):
      img1 = set1[i,:,:].copy()
      img2 = set2[i,:,:].copy()
      snr = psnr(img1, img2)
      psnr_avg.append(snr)
    return np.mean(psnr_avg)

def read_audio(filepath, sample_rate, normalize=True):
    """Read an audio file and return it as a numpy array"""
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
      div_fac = 1 / np.max(np.abs(audio)) / 3.0
      audio = audio * div_fac
    return audio, sr

def files_record(path,format_file='.WAV'):
    files = []
    name = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if format_file in file:
                files.append(os.path.join(r, file))
                name.append(file)
    return files

def add_noise_to_clean_audio(clean_audio, noise_signal):
    """Adds noise to an audio sample"""
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio

class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.hamming(self.window_length, sym=False)

    def get_stft_spectrogram(self):
        #return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            #window=self.window, center=True)
        return scipy.signal.stft(self.audio, nfft=self.ffT_length, nperseg=self.window_length, noverlap=self.overlap,
                            window=self.window, padded=True)
    
    def get_stft_spectrogram2(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)
        #return scipy.signal.stft(self.audio, nfft=self.ffT_length, nperseg=self.window_length, noverlap=self.overlap,
         #                   window=self.window, padded=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                             window=self.window, center=True)
    
    def get_audio_from_stft_spectrogram2(self, stft_features):
        return scipy.signal.istft(stft_features, nperseg=self.window_length, noverlap=self.overlap,
                             window=self.window)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                           n_fft=self.ffT_length, hop_length=self.overlap, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length, hop_length=self.overlap,
                                             win_length=self.window_length, window=self.window,
                                             center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)
    
    
def slice_stft(stft):
    mag = np.abs(stft[2])
    dt = mag.shape[1]
    if int(dt/numSegments)*numSegments!=dt:
        mag = mag[:,:int(dt/numSegments)*numSegments]
    split_stft = np.array_split(mag,int(mag.shape[1]/numSegments),axis=1)
    return split_stft

def segment_stack(stft):
    mag = np.abs(stft[2])
    sliced_image = slice_stft(stft)
    segments = np.zeros((mag.shape[0],numSegments,len(sliced_image)))
    for i in range(0,len(sliced_image)-1):
        stack = np.stack((sliced_image[i],sliced_image[i+1]),axis=-1)
        segments[:,:,i]= stack[:,:,0]
        if i+1 == len(sliced_image):
            segments[:,:,i+1]= stack[:,:,1]
    return segments

def built_dataset(filename):
    #dataset = np.empty((m,numFeatures,1))
    Audio, sr = read_audio(os.path.join(filename), sample_rate=fs)
    AudioFeatureExtractor = FeatureExtractor(Audio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
    stft_features = AudioFeatureExtractor.get_stft_spectrogram()
    seg = segment_stack(stft_features)
    dataset = seg.copy()
    #else:
     #   dataset = np.concatenate((dataset,seg),axis=2)
    return dataset