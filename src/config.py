# -*- coding: utf-8 -*-
import os


class Config:
    """
    A class that contains parameters and their default values for main.py and representation.py.

    Attributes:
        iterations (int): Number of iterations for training in main.py. Defaults to 3.
        plots (bool): Flag to determine if plots should be generated. Defaults to True.
        save_metrics (bool): Flag to determine if metrics should be saved. Defaults to True.
        load_checkpoint (bool): Flag to determine if the model checkpoint should be loaded. Defaults to True.
        album_split (bool): Flag to determine if album split should be used. Defaults to True.
        random_state (int): The seed for the random number generator. Used in representation.py. Defaults to 42.
        slice_length (int): The length of the slices for the songs. Used in representation.py. Defaults to 32.
        ensemble_visual (bool): Flag to determine if ensemble visualization should be used. Used in representation.py. Defaults to False.
        save_path (str): Path where to save representation outputs. Used in representation.py. Defaults to 'representation_output/'.
        nb_classes (int): Number of classes for the dataset. Defaults to 20.
        slice_lengths (list of int): Different lengths of slices to be used for the songs. Defaults to [911, 628, 313, 157, 94, 32].
        artist_folder (str): The directory containing artist data. Defaults to 'artists'.
        song_folder (str): The directory containing song data. Defaults to 'song_data'.
        train (bool): Flag to determine if model should be trained. Defaults to True.
        batch_size (int): The number of samples per gradient update. Defaults to 16.
        nb_epochs (int): Number of epochs to train the model. An epoch is an iteration over the entire dataset. Defaults to 200.
        early_stop (int): Number of no-improvement epochs to stop training. Defaults to 10.
        learning_rate (float): Learning rate for the model training. Defaults to 0.0001.
        random_state_list (list of int): List of seeds for the random number generator for different iterations. Defaults to [0, 21, 42].
        summary_metrics_output_folder (str): The directory to output summary metrics. Defaults depend on `album_split` value.
        save_metrics_folder (str): The directory to save metrics. Defaults depend on `album_split` value.
        save_weights_folder (str): The directory to save model weights. Defaults depend on `album_split` value.


        1s 32 frames
        3s 94 frames
        5s 157 frames
        6s 188 frames
        10s 313 frames
        20s 628 frames
        29.12s 911 frames

    """

    def __init__(self):
        # Set these parameters for main.py
        self.plots = True
        self.save_metrics = True
        self.load_checkpoint = True
        self.album_split = True

        # Set these parameters for representation.py and utility.py
        self.ensemble_visual = True
        self.wavelet = True

        # Do not change unless necessary
        self.nb_classes = 5
        self.slice_lengths = [32, 94, 157, 313, 628, 911]
        self.artist_folder = 'artists'
        self.train = True
        self.batch_size = 16
        self.nb_epochs = 200
        self.early_stop = 10
        self.learning_rate = 0.0001
        self.random_state_list = [0, 21, 42]

        if self.wavelet:
            self.song_folder = 'song_data_wavelet'
        else:
            self.song_folder = 'song_data'

        if self.album_split:
            self.summary_metrics_output_folder = os.path.join(
                'metrics', 'trials_album_split')
            self.save_metrics_folder = os.path.join(
                'metrics', 'metrics_album_split')
            self.save_weights_folder = os.path.join(
                'weights', 'weights_album_split')
            self.save_path = os.path.join(
                'representation_output', 'album_split')
        else:
            self.summary_metrics_output_folder = os.path.join(
                'metrics', 'trials_songs_split')
            self.save_metrics_folder = os.path.join(
                'metrics', 'metrics_songs_split')
            self.save_weights_folder = os.path.join(
                'weights', 'weights_songs_split')
            self.save_path = os.path.join(
                'representation_output', 'songs_split')


if __name__ == '__main__':
    owo = Config()
    print(
        f'summary_metrics_output_folder = {owo.summary_metrics_output_folder}')
    print(f'save_metrics_folder = {owo.save_metrics_folder}')
    print(f'save_weights_folder = {owo.save_weights_folder}')
    print(f'save_path = {owo.save_path}')
