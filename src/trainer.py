# -*- coding: utf-8 -*-
try:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
finally:
    import tensorflow as tf

from os.path import isfile
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from keras import Model, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import src.models as modelos
import src.utility as utility


class TrainingProcess:
    def __init__(self, nb_classes=20,
                 slice_length=911,
                 artist_folder='artists',
                 song_folder='song_data',
                 plots=True,
                 load_checkpoint=False,
                 save_metrics=True,
                 save_metrics_folder='metrics',
                 save_weights_folder='weights',
                 batch_size=16,
                 nb_epochs=200,
                 early_stop=10,
                 learning_rate=0.0001,
                 album_split=True,
                 random_states=42):
        self.nb_classes = nb_classes
        self.slice_length = slice_length
        self.artist_folder = artist_folder
        self.song_folder = song_folder
        self.plots = plots
        self.load_checkpoint = load_checkpoint
        self.save_metrics = save_metrics
        self.save_metrics_folder = save_metrics_folder
        self.save_weights_folder = save_weights_folder
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.album_split = album_split
        self.random_states = random_states

        self.weights, self.filename, self.matrix_filename, self.history_filename = self.initialize_directories()

    def set_visible_devices(self, device_type='GPU', indices=[0]):
        devices = tf.config.experimental.list_physical_devices(device_type)
        if devices:
            try:
                visible_devices = [devices[i] for i in indices]
                tf.config.experimental.set_visible_devices(
                    visible_devices, device_type)
            except RuntimeError as e:
                print(e)

    def set_memory_growth(self, device_type='GPU', allow_growth=True):
        devices = tf.config.experimental.list_physical_devices(device_type)
        if devices:
            try:
                for device in devices:
                    tf.config.experimental.set_memory_growth(
                        device, allow_growth)
            except RuntimeError as e:
                print(e)

    def initialize_directories(self):
        save_path_matrix = os.path.join(self.save_metrics_folder, 'matrix')
        save_path_history = os.path.join(self.save_metrics_folder, 'history')
        # weights\weights_songs|album_split
        os.makedirs(self.save_weights_folder, exist_ok=True)
        # metrics\metrics_songs|album_split\PNG
        os.makedirs(save_path_matrix, exist_ok=True)
        # metrics\metrics_songs|album_split\history
        os.makedirs(save_path_history, exist_ok=True)

        file_params = f"{self.nb_classes}_{self.slice_length}_{self.random_states}"

        # Pesos del modelo entrenado
        weights = os.path.join(self.save_weights_folder, f"{file_params}.hdf5")
        # Métricas del modelo entrenado
        filename = os.path.join(self.save_metrics_folder, file_params)
        # Confusion Matrix PNG
        matrix_filename = os.path.join(save_path_matrix, file_params)
        # History plot de las Epochs
        history_filename = os.path.join(save_path_history, file_params)

        return weights, filename, matrix_filename, history_filename

    def load_data(self):
        """
        Returns: sliced_dict(['X_train', 'Y_train', 'S_train', 'X_val', 'Y_val', 'S_val', 'X_test', 'Y_test', 'S_test'])\n
            X_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>\n
            Y_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>\n
            S_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>\n
            X_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>\n
            Y_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>\n
            S_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>\n
            X_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>\n
            Y_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>\n
            S_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
        """
        func = utility.load_dataset_album_split if self.album_split else utility.load_dataset_song_split
        datasets = func(song_folder_name=self.song_folder,
                        artist_folder=self.artist_folder,
                        nb_classes=self.nb_classes,
                        random_state=self.random_states)

        data_dict = dict(zip(['Y_train', 'X_train', 'S_train',
                              'Y_test', 'X_test', 'S_test',
                              'Y_val', 'X_val', 'S_val'], datasets))
        return data_dict

    def slice_data(self, data_dict: dict[str, Union[list[str], list[np.ndarray]]]) -> dict[str, Union[np.ndarray, str]]:
        """
        Returns: sliced_dict(['X_train', 'Y_train', 'S_train', 'X_val', 'Y_val', 'S_val', 'X_test', 'Y_test', 'S_test'])\n
            X_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>\n
            Y_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>\n
            S_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>\n
            X_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>\n
            Y_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>\n
            S_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>\n
            X_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>\n
            Y_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>\n
            S_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
        """
        sliced_data_dict = {}
        for key in ['train', 'val', 'test']:
            sliced_data_dict[f'X_{key}'], sliced_data_dict[f'Y_{key}'], sliced_data_dict[f'S_{key}'] = \
                utility.slice_songs(
                data_dict[f'X_{key}'],
                data_dict[f'Y_{key}'],
                data_dict[f'S_{key}'],
                length=self.slice_length)
        return sliced_data_dict

    def encode_and_reshape_data(self, data_dict) \
            -> tuple[dict[str, Union[np.ndarray, np.str_]], LabelEncoder, OneHotEncoder]:
        """
        Returns: encoded_data[0](['Y_train', 'X_train', 'S_train', 'Y_val', 'X_val', 'S_val', 'Y_test', 'X_test', 'S_test'])\n
            Y_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>\n
            X_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>\n
            S_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>\n
            Y_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>\n
            X_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>\n
            S_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>\n
            Y_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>\n
            X_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>\n
            S_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
        """
        reshaped_data_dict = {}
        le, enc = None, None  # Initialize label encoder and one-hot encoder

        # Iterate over train, val, and test sets
        for key in ['train', 'val', 'test']:
            # Encode the labels
            reshaped_data_dict[f'Y_{key}'], le, enc = utility.encode_labels(
                data_dict[f'Y_{key}'], le, enc)

            # Reshape the data
            reshaped_data_dict[f'X_{key}'] = data_dict[f'X_{key}'].reshape(
                data_dict[f'X_{key}'].shape + (1,))

            # Copy song slices
            reshaped_data_dict[f'S_{key}'] = data_dict[f'S_{key}']

        return reshaped_data_dict, le, enc

    def build_model(self, data_dict: dict[str, Union[np.ndarray, np.str_]], verboso=False):
        X_train_shape = data_dict['X_train'].shape  # <class 'tuple'>
        Y_train_shape = data_dict['Y_train'].shape[1]  # <class 'int'>
        if verboso:
            print(f"X_train_shape type: {type(X_train_shape)}, "
                  f"len={len(X_train_shape)}")
            print(f"Y_train_shape type: {type(Y_train_shape)}\n")

        # build the model
        model = modelos.CRNN2D(X_train_shape, nb_classes=Y_train_shape)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.summary()

        return model

    def load_weights(self, model: Model):
        # Initialize weights using checkpoint if it exists
        if self.load_checkpoint:
            print("Looking for previous weights...")
            if isfile(self.weights):
                print('Checkpoint file detected. Loading weights...\n')
                model.load_weights(self.weights)
            else:
                print('No checkpoint file detected.  Starting from scratch.\n')
        else:
            print('Starting from scratch (no checkpoint)\n')

        # Customize the setting for model training.
        checkpointer = ModelCheckpoint(filepath=self.weights,
                                       verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                                     patience=self.early_stop, verbose=0, mode='auto')
        return checkpointer, earlystopper

    def train_model(self, model: Model, checkpointer: ModelCheckpoint, earlystopper: EarlyStopping,
                    data_dict: dict[str, Union[np.ndarray, np.str_]]):
        X_train = data_dict['X_train']
        Y_train = data_dict['Y_train']
        X_val = data_dict['X_val']
        Y_val = data_dict['Y_val']

        print("Input Data Shape", X_train.shape)
        history = model.fit(X_train, Y_train, batch_size=self.batch_size,
                            shuffle=True, epochs=self.nb_epochs,
                            verbose=1, validation_data=(X_val, Y_val),
                            callbacks=[checkpointer, earlystopper])

        if self.plots:
            utility.plot_history(history=history, history_filename=self.history_filename,
                                 title="Model accuracy", ext="pdf")

    def score_model(self, model: Model, data_dict: dict[str, Union[np.ndarray, np.str_]], le: LabelEncoder, ext='png'):
        # data_dict: tuple[dict, LabelEncoder, OneHotEncoder]
        X_train = data_dict['X_train']  # <class 'numpy.numpy.ndarray'>
        X_test = data_dict['X_test']  # <class 'numpy.numpy.ndarray'>
        Y_test = data_dict['Y_test']  # <class 'numpy.numpy.ndarray'>
        S_test = data_dict['S_test']  # <class 'numpy.str_'>

        # Load weights that gave best performance on validation set
        model.load_weights(self.weights)

        # Score test model
        score = model.evaluate(X_test, Y_test, verbose=0)
        y_score = model.predict(X_test)
        class_names = np.arange(self.nb_classes)
        class_names_original = le.inverse_transform(class_names)
        print(f"class_names = {class_names}")
        print(f"class_names_original = {class_names_original}")

        # Print out metrics
        print('Test score/loss:', score[0])
        print('Test accuracy:', score[1])
        print('\nTest results on each slice:')

        # Predict artist by a single frame
        y_true = np.argmax(Y_test, axis=1)
        y_predict = np.argmax(y_score, axis=1)

        scores_str = classification_report(y_true=y_true, y_pred=y_predict,
                                           target_names=class_names_original,
                                           zero_division=0)
        scores_dict = classification_report(y_true=y_true, y_pred=y_predict,
                                            target_names=class_names_original,
                                            output_dict=True, zero_division=0)
        print(scores_str)

        # Predict artist using pooling methodology
        actual_array, prediction_array = utility.predict_artist(model, X_test, Y_test, S_test,
                                                                le, slices=None, verbose=False)
        pooling_scores_str = classification_report(y_true=actual_array, y_pred=prediction_array,
                                                   target_names=class_names_original, zero_division=0)
        pooled_scores_dict = classification_report(y_true=actual_array, y_pred=prediction_array,
                                                   target_names=class_names_original, output_dict=True, zero_division=0)
        print(pooling_scores_str)

        if self.save_metrics:
            # Plot the confusion matrix (frame level)
            A = confusion_matrix(y_true, y_predict)
            utility.plot_confusion_matrix(A, classes=class_names_original, normalize=True,
                                          file_path=f"{self.matrix_filename}.{ext}",
                                          title='Confusion matrix')
            del A, y_true, y_predict

            B = confusion_matrix(actual_array, prediction_array)
            utility.plot_confusion_matrix(B, classes=class_names_original, normalize=True,
                                          file_path=f"{self.matrix_filename}_pooled.{ext}",
                                          title='Confusion matrix for pooled results')
            del B, actual_array, prediction_array

            # Save metrics
            with open(f"{self.filename}.txt", 'w') as f:
                f.write(f"Training data shape: {X_train.shape}\n")
                f.write(f"nb_classes: {self.nb_classes}\n")
                f.write(f"slice_length: {self.slice_length}\n")
                f.write(f"weights: {self.weights}\n")
                f.write(f"learning_rate: {self.learning_rate}\n")
                f.write(f"Test score/loss: {score[0]}\n")
                f.write(f"Test accuracy: {score[1]}\n")
                f.write("Test results on each slice:\n")
                f.write(f"{scores_str}\n")
                f.write("\n Scores when pooling song slices:\n")
                f.write(f"{pooling_scores_str}\n")

        return (scores_dict, pooled_scores_dict)
