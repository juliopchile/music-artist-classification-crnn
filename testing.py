import os

import cv2
import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pywt


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# *****************
# ** Signal Audio**
# *****************
song_path = 'artists/metallica/Kill_Em_All/01-Hit_The_Lights.mp3'
t0 = 117
duration = 10
rate = 16000
waveletname = 'morl'
dt = 1/rate

signal, sr = librosa.load(path=song_path, sr=rate,
                          offset=t0, duration=duration)

# ****************
# ** Signal Test**
# ****************
# Define some parameters
sr = 16000  # samples per second
freqs = [512, 1024, 2048, 4096]  # A4, A5, A6 in Hz

# Define parameters for the chirp
f0 = 20.0  # start frequency in Hz
f1 = 8000.0  # end frequency in Hz

# Generate the time axis
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# signal = np.sum([np.sin(2 * np.pi * freq * t) for freq in freqs], axis=0)
# signal /= np.max(np.abs(signal))

# Generate the chirp signal
signal = scipy.signal.chirp(t, f0, duration, f1, method='linear')

# *********************
# ** Mel Spectrogram **
# *********************
n_fft = 2048
hop_length = 512

S = librosa.feature.melspectrogram(
    y=signal, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
log_S = librosa.power_to_db(S, ref=1.0)

D = librosa.stft(signal)  # compute STFT of the signal
D_db = librosa.amplitude_to_db(abs(D))  # convert to dB

alto = log_S.shape[0]
ancho = log_S.shape[1]

# Calculate the time corresponding to each frame
times = np.linspace(0, duration, ancho)

# ***********************************
# ** Wavelet Scalogram Frequencies **
# ***********************************
# For scales we want to focus at certain frequencies, since this is a Music Information Retrieval task (MIR),
# we use the mel scale and select the frequencies from 20 to 8000 as our relevant ones.
# 20 since is the lower humans can hear and 8000 is the nyquist frequency of our audio.
freq_num = 128

# Compute the mel values spaced between the choosen range
low_freq, high_freq = 20, sr/2
low_mel, high_mel = librosa.hz_to_mel(low_freq), librosa.hz_to_mel(high_freq)

mel_freq1 = np.linspace(low_mel, high_mel, num=(freq_num))
mel_freq2 = np.geomspace(low_mel, high_mel, num=(freq_num))

# Compute the corresponding frequencies
frequencies1 = np.linspace(low_freq, high_freq, num=(freq_num))
frequencies2 = np.geomspace(low_freq, high_freq, num=(freq_num))
frequencies3 = librosa.mel_to_hz(mel_freq1)
frequencies4 = librosa.mel_to_hz(mel_freq2)

# Compute the corresponding scales for our specific frequencies
scales1 = pywt.frequency2scale(wavelet=waveletname, freq=frequencies1)*rate
scales2 = pywt.frequency2scale(wavelet=waveletname, freq=frequencies2)*rate
scales3 = pywt.frequency2scale(wavelet=waveletname, freq=frequencies3)*rate
scales4 = pywt.frequency2scale(wavelet=waveletname, freq=frequencies4)*rate


# **********************************************
# **Computing CWT (continuous wavelet transform)
# **********************************************
coef1, freq1 = pywt.cwt(data=signal, scales=scales1, wavelet=waveletname,
                        sampling_period=dt, method='fft', axis=-1)
coef2, freq2 = pywt.cwt(data=signal, scales=scales2, wavelet=waveletname,
                        sampling_period=dt, method='fft', axis=-1)
coef3, freq3 = pywt.cwt(data=signal, scales=scales3, wavelet=waveletname,
                        sampling_period=dt, method='fft', axis=-1)
coef4, freq4 = pywt.cwt(data=signal, scales=scales4, wavelet=waveletname,
                        sampling_period=dt, method='fft', axis=-1)

scalogram1 = librosa.amplitude_to_db(np.abs(coef1))
scalogram2 = librosa.amplitude_to_db(np.abs(coef2))
scalogram3 = librosa.amplitude_to_db(np.abs(coef3))
scalogram4 = librosa.amplitude_to_db(np.abs(coef4))

# Resize to match Mel spectrogram dimensions
resized1 = cv2.resize(scalogram1, (ancho, alto), interpolation=cv2.INTER_AREA)
resized2 = cv2.resize(scalogram2, (ancho, alto), interpolation=cv2.INTER_AREA)
resized3 = cv2.resize(scalogram3, (ancho, alto), interpolation=cv2.INTER_AREA)
resized4 = cv2.resize(scalogram4, (ancho, alto), interpolation=cv2.INTER_AREA)

mel_freq1 = np.linspace(low_mel, high_mel, num=alto)
mel_freq2 = np.geomspace(low_mel, high_mel, num=alto)
frequencies1 = np.linspace(low_freq, high_freq, num=alto)
frequencies2 = np.geomspace(low_freq, high_freq, num=alto)
frequencies3 = librosa.mel_to_hz(mel_freq1)
frequencies4 = librosa.mel_to_hz(mel_freq2)


# **********
# ** Plot **
# **********
cmap = 'magma'
interpolation = 'none'

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Plot the STFT spectrogram
img = librosa.display.specshow(D_db, sr=sr, ax=axs[0, 0], cmap=cmap,
                               y_axis='fft', x_axis='s', auto_aspect='true')
axs[0, 0].set_title('STFT Spectrogram (db)')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Frequency [Hz]')

# Plot the Mel spectrogram
img = librosa.display.specshow(log_S, sr=sr, ax=axs[1, 0], cmap=cmap,
                               y_axis='mel', x_axis='s', auto_aspect='true')
axs[1, 0].set_title('Mel Spectrogram (db)')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Frequency [Hz]')

resize = True

# Prepare a list of parameters for the scalograms
if resize:
    data = [
        ('Power Scalogram (db): Linear Space', resized1, frequencies1),
        ('Power Scalogram (db): Geometric Space', resized2, frequencies2),
        ('Power Scalogram (db): Mel-to-Hz (Linear Mel Space)', resized3, frequencies3),
        ('Power Scalogram (db): Mel-to-Hz (Geometric Mel Space)', resized4, frequencies4)
    ]
else:
    data = [
        ('Power Scalogram (db): Linear Space', scalogram1, freq1),
        ('Power Scalogram (db): Geometric Space', scalogram2, freq2),
        ('Power Scalogram (db): Mel-to-Hz (Linear Mel Space)', scalogram3, freq3),
        ('Power Scalogram (db): Mel-to-Hz (Geometric Mel Space)', scalogram4, freq4)
    ]

ytick_freqs = [0, 512, 1024, 2048, 4096]
# Loop over the axes and parameters together. Skip the first subplot
for ax, (title, image, frequencies) in zip(axs[[0, 0, 1, 1], [1, 2, 1, 2]].flatten(), data):
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
