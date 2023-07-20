import os

import cv2
import librosa
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pywt
import time


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def is_continuous_wavelet(wavelet):
    try:
        wavelet_obj = pywt.ContinuousWavelet(wavelet)
        return True
    except ValueError:
        return False


def wavelet_transform(signal, scales, wavelet_name, sampling_period):
    # Check if the wavelet is continuous or discrete
    if is_continuous_wavelet(wavelet_name):
        # Call the cwt function if wavelet is continuous
        coef, _ = pywt.cwt(data=signal, scales=scales, wavelet=wavelet_name,
                           sampling_period=sampling_period, method='fft', axis=-1)
    else:
        # Call the dwt function if wavelet is discrete
        # Note: scales are not used in dwt and the wavelet_name must be a Wavelet object
        wavelet = pywt.Wavelet(wavelet_name)
        coef, freqs = pywt.dwt(data=signal, wavelet=wavelet, mode='reflect')

    return coef


# *****************
# ** Signal Audio**
# *****************
song_path = 'artists/u2/The_Joshua_Tree/02-I_Still_Haven_t_Found_What_I_m_Looking_For.mp3'
t0 = 0
duration = None
rate = 16000

waveletname1 = 'fbsp3-0.07-1.0'
waveletname4 = 'morl'
waveletname2 = 'shan0.1-2.0'
waveletname5 = 'shan0.1-1.7'
waveletname3 = 'cmor2.0-8.0'
waveletname6 = 'cmor2.5-8.0'

dt = 1/rate

# candidatos: ['fbsp3-0.07-1.0', 'shan0.1-2.0', 'cmor2.5-8.0'

senal, sr = librosa.load(path=song_path, sr=rate,
                         offset=t0, duration=duration)

# ****************
# ** Signal Test**
# ****************
# Define some parameters
sr = 16000  # samples per second
freqs = [128, 256, 512, 1024, 2048, 4096, 8000]

# Generate the time axis
# t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# senal = np.sum([np.sin(2 * np.pi * freq * t) for freq in freqs], axis=0)
# senal = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
# senal /= np.max(np.abs(senal))

# Generate the chirp signal
# senal = scipy.signal.chirp(t, 1, duration, 8000, method='linear')

# *********************
# ** Mel Spectrogram **
# *********************
n_fft = 2048
hop_length = 512

S = librosa.feature.melspectrogram(
    y=senal, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
log_S = librosa.power_to_db(S, ref=1.0)

alto = log_S.shape[0]
ancho = log_S.shape[1]

# ***********************************
# ** Wavelet Scalogram Frequencies **
# ***********************************
# For scales we want to focus at certain frequencies, since this is a Music Information Retrieval task (MIR),
# we use the mel scale and select the frequencies from 20 to 8000 as our relevant ones.
# 20 since is the lower humans can hear and 8000 is the nyquist frequency of our audio.
freq_num = 128
resize = False

# Compute the mel values spaced between the choosen range
low_freq, high_freq = 32, 8000
low_mel, high_mel = librosa.hz_to_mel(low_freq), librosa.hz_to_mel(high_freq)

mel_freq1 = np.linspace(low_mel, high_mel, num=(freq_num))
mel_freq2 = np.geomspace(low_mel, high_mel, num=(freq_num))

# Compute the corresponding frequencies
frequencies1 = np.linspace(low_freq, high_freq, num=(freq_num))
frequencies2 = np.geomspace(low_freq, high_freq, num=(freq_num))
frequencies3 = librosa.mel_to_hz(mel_freq1)
frequencies4 = librosa.mel_to_hz(mel_freq2)

freq_to_use = frequencies3

# Compute the corresponding scales for our specific frequencies
scales1 = pywt.frequency2scale(wavelet=waveletname1, freq=freq_to_use)*rate
scales2 = pywt.frequency2scale(wavelet=waveletname2, freq=freq_to_use)*rate
scales3 = pywt.frequency2scale(wavelet=waveletname3, freq=freq_to_use)*rate
scales4 = pywt.frequency2scale(wavelet=waveletname4, freq=freq_to_use)*rate
scales5 = pywt.frequency2scale(wavelet=waveletname5, freq=freq_to_use)*rate
scales6 = pywt.frequency2scale(wavelet=waveletname6, freq=freq_to_use)*rate


# **********************************************
# **Computing CWT (continuous wavelet transform)
# **********************************************
start_time = time.time()
coef1 = wavelet_transform(senal, scales1, waveletname1, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname1}: {elapsed_time} seconds")

start_time = time.time()
coef2 = wavelet_transform(senal, scales2, waveletname2, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname2}: {elapsed_time} seconds")

start_time = time.time()
coef3 = wavelet_transform(senal, scales3, waveletname3, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname3}: {elapsed_time} seconds")

start_time = time.time()
coef4 = wavelet_transform(senal, scales4, waveletname4, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname4}: {elapsed_time} seconds")

start_time = time.time()
coef5 = wavelet_transform(senal, scales5, waveletname5, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname5}: {elapsed_time} seconds")

start_time = time.time()
coef6 = wavelet_transform(senal, scales6, waveletname6, dt)
elapsed_time = time.time() - start_time
print(f"Time taken for wavelet {waveletname6}: {elapsed_time} seconds")

scalogram1 = librosa.amplitude_to_db(np.abs(coef1))
scalogram2 = librosa.amplitude_to_db(np.abs(coef2))
scalogram3 = librosa.amplitude_to_db(np.abs(coef3))
scalogram4 = librosa.amplitude_to_db(np.abs(coef4))
scalogram5 = librosa.amplitude_to_db(np.abs(coef5))
scalogram6 = librosa.amplitude_to_db(np.abs(coef6))

# Resize to match Mel spectrogram dimensions
resized1 = cv2.resize(scalogram1, (ancho, alto), interpolation=cv2.INTER_AREA)
resized2 = cv2.resize(scalogram2, (ancho, alto), interpolation=cv2.INTER_AREA)
resized3 = cv2.resize(scalogram3, (ancho, alto), interpolation=cv2.INTER_AREA)
resized4 = cv2.resize(scalogram4, (ancho, alto), interpolation=cv2.INTER_AREA)
resized5 = cv2.resize(scalogram5, (ancho, alto), interpolation=cv2.INTER_AREA)
resized6 = cv2.resize(scalogram6, (ancho, alto), interpolation=cv2.INTER_AREA)

mel_freq1 = np.linspace(low_mel, high_mel, num=alto)
mel_freq2 = np.geomspace(low_mel, high_mel, num=alto)
freq_resized1 = np.linspace(low_freq, high_freq, num=alto)
freq_resized2 = np.geomspace(low_freq, high_freq, num=alto)
freq_resized3 = librosa.mel_to_hz(mel_freq1)
freq_resized4 = librosa.mel_to_hz(mel_freq2)

freq_resize = freq_resized3


# **********
# ** Plot **
# **********

cmap = 'magma'
interpolation = 'none'

fig, axs = plt.subplots(2, 3, figsize=(20, 10))


# Prepare a list of parameters for the scalograms
if resize:
    data = [
        (f'Wavelet: {waveletname1}', resized1, freq_resize),
        (f'Wavelet: {waveletname2}', resized2, freq_resize),
        (f'Wavelet: {waveletname3}', resized3, freq_resize),
        (f'Wavelet: {waveletname4}', resized4, freq_resize),
        (f'Wavelet: {waveletname5}', resized5, freq_resize),
        (f'Wavelet: {waveletname6}', resized6, freq_resize)
    ]

else:
    data = [
        (f'Wavelet: {waveletname1}', scalogram1, freq_to_use),
        (f'Wavelet: {waveletname2}', scalogram2, freq_to_use),
        (f'Wavelet: {waveletname3}', scalogram3, freq_to_use),
        (f'Wavelet: {waveletname4}', scalogram4, freq_to_use),
        (f'Wavelet: {waveletname5}', scalogram5, freq_to_use),
        (f'Wavelet: {waveletname6}', scalogram6, freq_to_use)
    ]

ytick_freqs = [32, 128, 256, 512, 1024, 2048, 4096, 8000]
# Loop over the axes and parameters together. Skip the first subplot
for ax, (title, image, frequencies) in zip(axs.flatten(), data):
    ax.imshow(image, aspect='auto', cmap=cmap,
              interpolation=interpolation, origin='lower')

    ytick_idxs = [find_nearest_idx(frequencies, freq) for freq in ytick_freqs]
    ax.set_yticks(ytick_idxs)
    ax.set_yticklabels(ytick_freqs)

    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()
