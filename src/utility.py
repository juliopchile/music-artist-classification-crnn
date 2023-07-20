# -*- coding: utf-8 -*-
import itertools
import os
import random

import cv2
import dill
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from keras.callbacks import History
from matplotlib import cm
from numpy.random import RandomState
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from tqdm import tqdm as tqdm

from src.config import Config


class NoRateTqdm(tqdm):
    def format_meter(self, *args, **kwargs):
        kwargs['bar_format'] = '{desc}: {l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]'
        return super(NoRateTqdm, self).format_meter(*args, **kwargs)


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
        coef, _ = pywt.dwt(data=signal, wavelet=wavelet)

    return coef


def get_mel_spectrogram(path: str, sr=16000, n_mels=128, n_fft=2048, hop_length=512, duration=None, offset=0):
    """
    Compute Mel-scaled power spectrogram from an audio file.

    Args:
    path (str): Path to the audio file.
    sr (int): Sampling rate for audio processing.
    n_mels (int): Number of Mel bands to generate.
    n_fft (int): Length of the FFT window.
    hop_length (int): Number of samples between successive frames.
    duration (float): Duration of the audio file, in seconds.
    offset (float): Start reading after this time (in seconds).

    Returns:
    np.ndarray: Mel-scaled spectrogram in decibels (dB).
    """
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    return librosa.power_to_db(S, ref=1.0)


def get_wavelet_scalogram(path: str, sr=16000, n_mels=128, n_fft=2048, hop_length=512, duration=None, offset=0, waveletname='shan0.1-1.7'):
    # Obtain signal
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)

    # Obtain melspectrogram to check length (used for the wavelet scalogram resample)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    ancho, alto = S.shape[1], S.shape[0]
    del S

    # Define the frequencies an scales of the wavelet transform
    low_freq, high_freq = 20, sr/2
    mel_freq = np.linspace(librosa.hz_to_mel(low_freq),
                           librosa.hz_to_mel(high_freq),
                           num=128)
    frequencies = librosa.mel_to_hz(mel_freq)
    scales = pywt.frequency2scale(wavelet=waveletname, freq=frequencies)*sr
    dt = 1/sr

    # Compute the Wavelet Transform
    coef = wavelet_transform(y, scales, waveletname, dt)
    scalogram = librosa.amplitude_to_db(np.abs(coef))

    # Downsample to fit the same dimensions as mel spectrogram
    resized = cv2.resize(scalogram, (ancho, alto),
                         interpolation=cv2.INTER_AREA)

    return resized


def print_names(names: list, num_columns: int):
    """
    Print formatted artist names inside a box.

    Args:
    names (list): List of artist names.
    num_columns (int): Number of columns to use when printing.
    """
    # The width of the formatted string
    name_width = max(len(name) for name in names) + 2

    # Add padding to each name and format into columns
    formatted_names = [f"{name:<{name_width}}" for name in names]

    # Form rows
    rows = [formatted_names[i:i+num_columns]
            for i in range(0, len(formatted_names), num_columns)]
    rows_str = ['  '.join(row) for row in rows]

    # Determine the width of the longest row
    row_width = max(len(row) for row in rows_str)

    # Print the box top
    # 8 extra for '**  ' at beginning and '  **' at end
    print('*' * (row_width + 8))
    print("**{: ^{}}**".format('CREATING DATA SET INSIDE \'SONG_DATA\' FOLDER', row_width + 4))
    print('*' * (row_width + 8))

    # Print the formatted artist names
    for row_str in rows_str:
        print(f'**  {row_str:<{row_width}}  **')

    # Print the box bottom
    print('*' * (row_width + 8))

    return None


def create_dataset(artist_folder='artists', save_folder='song_data', sr=16000, n_mels=128, n_fft=2048, hop_length=512, wavelet_rep=False):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(os.path.join(artist_folder, path))]

    # Print informative box
    print_names(artists, num_columns=4)

    # Get total number of songs for progress bar
    total_songs = 0
    for artist in artists:
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)
            total_songs += len(album_songs)

    pbar_artists = tqdm(total=total_songs, desc="Procesando")

    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in artists:
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)

            for song in album_songs:
                song_path = os.path.join(album_path, song)

                if wavelet_rep:
                    log_S = get_wavelet_scalogram(
                        path=song_path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                else:
                    # Create mel spectrogram and convert it to the log scale
                    log_S = get_mel_spectrogram(
                        path=song_path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

                data = (artist, log_S, song)

                # Save each song
                save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)

                # Update the progress bar
                pbar_artists.update()
                pbar_artists.set_postfix_str(
                    f'{artist}/{album}/{song}', refresh=True)

    # Remove the progress bar
    pbar_artists.close()


def load_dataset(song_folder_name: str = 'song_data', artist_folder: str = 'artists', nb_classes: int = 20,
                 random_state: int = 42) -> tuple[list[str], list[np.ndarray], list[str]]:
    """
    This function loads the dataset based on a location;
    it returns a tuple with lists of spectrograms and their corresponding artists/song names
    """

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        # loaded_song: <class 'tuple'>, length: 3
        # loaded_song[0]: artist <class 'str'>
        # loaded_song[1]: spectrogram <class 'numpy.ndarray'> lenght = 128
        # loaded_song[2]: number-songname.mp3 <class 'str'>
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name


def load_dataset_album_split(song_folder_name='song_data', artist_folder='artists', nb_classes=20, random_state=42) \
        -> tuple[list[str], list[np.ndarray], list[str], list[str], list[np.ndarray], list[str], list[str], list[np.ndarray], list[str]]:
    """ This function loads a dataset and splits it on an album level"""

    # Load the list of song and artist
    song_list = os.listdir(song_folder_name)
    artist_list = os.listdir(artist_folder)

    train_albums = []
    test_albums = []
    val_albums = []

    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        albums = os.listdir(os.path.join(artist_folder, artist))
        random.shuffle(albums)
        test_albums.append(artist + '_%%-%%_' + albums.pop(0))
        val_albums.append(artist + '_%%-%%_' + albums.pop(0))
        train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        # loaded_song: <class 'tuple'>, length: 3
        # loaded_song[0]: artist <class 'str'>
        # loaded_song[1]: no se <class 'numpy.ndarray'> lenght = variable
        # loaded_song[2]: number-songname.mp3 <class 'str'>
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        artist, album, song_name = song.split('_%%-%%_')
        artist_album = artist + '_%%-%%_' + album

        if loaded_song[0] in artists:
            if artist_album in train_albums:
                Y_train.append(loaded_song[0])
                X_train.append(loaded_song[1])
                S_train.append(loaded_song[2])
            elif artist_album in test_albums:
                Y_test.append(loaded_song[0])
                X_test.append(loaded_song[1])
                S_test.append(loaded_song[2])
            elif artist_album in val_albums:
                Y_val.append(loaded_song[0])
                X_val.append(loaded_song[1])
                S_val.append(loaded_song[2])

    return Y_train, X_train, S_train, \
        Y_test, X_test, S_test, \
        Y_val, X_val, S_val


def load_dataset_song_split(song_folder_name='song_data',
                            artist_folder='artists',
                            nb_classes=20,
                            test_split_size=0.1,
                            validation_split_size=0.1,
                            random_state=42) \
        -> tuple[list[str], list[np.ndarray], list[str], list[str], list[np.ndarray], list[str], list[str], list[str], list[str]]:

    Y, X, S = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        X, Y, S, test_size=test_split_size, stratify=Y,
        random_state=random_state)

    # X_train: <class 'list'>  len=1271, interior type=<class 'numpy.ndarray'>
    # X_test:  <class 'list'>   len=142, interior type=<class 'numpy.ndarray'>
    # Y_train: <class 'list'>  len=1271, interior type=<class 'str'>
    # Y_test:  <class 'list'>   len=142, interior type=<class 'str'>
    # S_train: <class 'list'>  len=1271, interior type=<class 'str'>
    # S_test:  <class 'list'>   len=142, interior type=<class 'str'>

    # Create a validation to be used to track progress
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(
        X_train, Y_train, S_train, test_size=validation_split_size,
        shuffle=True, stratify=Y_train, random_state=random_state)

    # X_train: <class 'list'>    len=1143, interior type=<class 'numpy.ndarray'>
    # X_test:  <class 'list'>     len=142, interior type=<class 'numpy.ndarray'>
    # Y_train: <class 'list'>    len=1143, interior type=<class 'str'>
    # Y_test:  <class 'list'>     len=142, interior type=<class 'str'>
    # S_train: <class 'list'>    len=1143, interior type=<class 'str'>
    # S_test:  <class 'list'>     len=142, interior type=<class 'str'>

    return Y_train, X_train, S_train, \
        Y_test, X_test, S_test, \
        Y_val, X_val, S_val


def slice_songs(X, Y, S, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(Y[i])
            song_name.append(S[i])

    return np.array(spectrogram), np.array(artist), np.array(song_name)


def create_single_spectrogram_plot(path: str, duration=None, offset=0, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Create a Mel-scaled power spectrogram for an audio file and saves it.

    Args:
    path (str): Path to the audio file.
    duration (float): Duration of the audio file, in seconds.
    offset (float): Start reading after this time (in seconds).
    sr (int): Sampling rate for audio processing.
    n_mels (int): Number of Mel bands to generate.
    n_fft (int): Length of the FFT window.
    hop_length (int): Number of samples between successive frames.
    """

    # Make a mel-scaled power (energy-squared) spectrogram
    log_S = get_mel_spectrogram(
        path, sr, n_mels, n_fft, hop_length, duration, offset)

    # Save the single spectrogram as a PDF
    fig, ax = plt.subplots(figsize=(12, 5))

    img = librosa.display.specshow(log_S, sr=sr, ax=ax)

    ax.set_title('mel power spectrogram')
    fig.colorbar(img, format='%+02.0f dB', ax=ax)

    ax.set_ylabel('Mel scale')
    ax.set_xlabel('Time')

    fig.tight_layout()
    fig.savefig("One_spectrogram.pdf", bbox_inches="tight")
    plt.close(fig)

    return None


def create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                             n_fft=2048, hop_length=512):
    """Create a spectrogram from a randomly selected song
     for each artist and plot"""

    # get list of all artists
    artists = os.listdir(artist_folder)

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(
        14, 12), sharex=True, sharey=True)

    # iterate through artists, randomly select an album,
    # randomly select a song, and plot a spectrogram on a grid
    for idx, artist in enumerate(artists):
        # Randomly select album and song
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        album = random.choice(artist_albums)
        album_path = os.path.join(artist_path, album)
        album_songs = os.listdir(album_path)
        song = random.choice(album_songs)
        song_path = os.path.join(album_path, song)

        # Create mel spectrogram
        log_S = get_mel_spectrogram(
            song_path, sr, n_mels, n_fft, hop_length, 3, 60)

        # Plot on grid
        ax = axs[idx // 5, idx % 5]
        img = librosa.display.specshow(log_S, sr=sr, ax=ax)
        ax.set_title(artist)
        fig.colorbar(img, ax=ax)

    fig.tight_layout()
    return fig, axs


def plot_confusion_matrix(cm, classes, normalize=False, file_path='matrix.png',
                          title='Confusion matrix', cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'{title}, with normalization'

    fig, ax = plt.subplots(figsize=(14, 14))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return None


def plot_history(history: History, history_filename: str, title="model accuracy", ext="png"):
    """
    This function plots the training and validation accuracy
    per epoch of training
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # Add label within the plot method
    ax.plot(history.history['accuracy'], label='train')
    # Changed 'test' to 'validation' for clarity
    ax.plot(history.history['val_accuracy'], label='validation')

    ax.set_title(title)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    # Legends are automatically filled from the label argument of plot method
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(f"{history_filename}.{ext}", bbox_inches="tight")
    plt.close(fig)

    return None


def predict_artist(model, X: np.ndarray, Y: np.ndarray, S: np.str_, le: LabelEncoder, slices=None, verbose=False, ml_mode=False):
    """ This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    print("Test results when pooling slices by song and voting:")
    # Obtain the list of songs
    songs = np.unique(S)
    prediction_list = []
    actual_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        X_song = X[S == song]
        Y_song = Y[S == song]
        if verbose:
            print(f"X_song.shape = {X_song.shape}")
            print(f"Y_song.shape = {X_song.shape}")

        # If not using full song, shuffle and take up to a number of slices
        if slices and slices <= X_song.shape[0]:
            X_song, Y_song = shuffle(X_song, Y_song)
            X_song = X_song[:slices]
            Y_song = Y_song[:slices]

        # Get probabilities of each class
        predictions = model.predict(X_song, verbose=0)

        if not ml_mode:
            # Get list of highest probability classes and their probability
            class_prediction = np.argmax(predictions, axis=1)
            class_probability = np.max(predictions, axis=1)

            # keep only predictions confident about;
            prediction_summary_trim = class_prediction[class_probability > 0.5]

            # deal with edge case where there is no confident class
            if len(prediction_summary_trim) == 0:
                prediction_summary_trim = class_prediction
        else:
            prediction_summary_trim = predictions

        try:
            prediction = stats.mode(prediction_summary_trim)[0][0]
            actual = stats.mode(np.argmax(Y_song))[0][0]
        except:
            prediction = stats.mode(prediction_summary_trim)[0]
            actual = stats.mode(np.argmax(Y_song))[0]

        # Keeping track of overall song classification accuracy
        prediction_list.append(prediction)
        actual_list.append(actual)

        # Print out prediction
        if verbose:
            print(song)
            print("Predicted:", le.inverse_transform([prediction]), "\nActual:",
                  le.inverse_transform([actual]))
            print('\n')

    # Print overall song accuracy
    actual_array = np.array(actual_list)
    prediction_array = np.array(prediction_list)

    return (actual_array, prediction_array)


def encode_labels(Y, le=None, enc=None) -> tuple[np.ndarray, LabelEncoder, OneHotEncoder]:
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = LabelEncoder()
        Y_le = le.fit_transform(Y).reshape(N, 1)
    else:
        Y_le = le.transform(Y).reshape(N, 1)

    # convert into one hot encoding
    if enc is None:
        enc = OneHotEncoder()
        Y_enc = enc.fit_transform(Y_le).toarray()
    else:
        Y_enc = enc.transform(Y_le).toarray()

    # return encoders to re-use on other data
    return Y_enc, le, enc


def simple_encoding(Y, le=None):
    """Encodes target variables into numbers"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y)
    else:
        Y_le = le.transform(Y)

    # return encoders to re-use on other data
    return Y_le, le


if __name__ == '__main__':
    cfg = Config()

    # configuration options
    create_data = True
    create_visuals = False
    save_visuals = False

    if create_data:
        create_dataset(artist_folder='artists', save_folder=cfg.song_folder,
                       sr=16000, n_mels=128, n_fft=2048,
                       hop_length=512, wavelet_rep=cfg.wavelet)

    if create_visuals:
        # Create spectrogram for a specific song
        create_single_spectrogram_plot(
            'artists/u2/The_Joshua_Tree/' +
            '02-I_Still_Haven_t_Found_What_I_m_Looking_For.mp3',
            offset=60, duration=29.12)

        # Create spectrogram subplots
        fig, axs = create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                                            n_fft=2048, hop_length=512)
        if save_visuals:
            fig.savefig(os.path.join('Spectrograms.pdf'),
                        bbox_inches="tight")
        plt.close(fig)
