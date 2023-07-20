# -*- coding: utf-8 -*-
try:
    import os
    from os.path import isfile
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
finally:
    import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gc

from tqdm import tqdm
from keras.optimizers import Adam
from sklearn.manifold import TSNE

import src.utility as utility
import src.models as models
from src.config import Config


if __name__ == '__main__':
    # with tf.device('/CPU:0'):
    # Instantiate the Config class
    cfg = Config()

    # leave as-is
    slice_length = 157
    # random_state = 0
    nb_classes = cfg.nb_classes
    folder = cfg.song_folder
    learning_rate = cfg.learning_rate
    for ensemble_visual in [True, False]:
        for random_state in cfg.random_state_list:

            checkpoint_path = os.path.join(cfg.save_weights_folder, str(nb_classes) +
                                           '_' + str(slice_length) + '_' + str(random_state) + '.hdf5')

            # Load the song data and split into train and test sets at song level
            print(f"Loading data for {slice_length}_{random_state}")
            print(f"checkpoint_path = {checkpoint_path}")

            Y, X, S = utility.load_dataset(song_folder_name=folder,
                                           nb_classes=nb_classes,
                                           random_state=random_state)
            X, Y, S = utility.slice_songs(X, Y, S, length=slice_length)

            # Reshape data as 2d convolutional tensor shape
            X_shape = X.shape + (1,)
            X = X.reshape(X_shape)

            # encode Y
            Y_original = Y
            Y, le, enc = utility.encode_labels(Y)

            # build the model
            model = models.CRNN2D(X.shape, nb_classes=Y.shape[1])
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=cfg.learning_rate),
                          metrics=['accuracy'])

            # Initialize weights using checkpoint if it exists
            if isfile(checkpoint_path):
                print('Checkpoint file detected. Loading weights.')
                model.load_weights(checkpoint_path)
            else:
                raise Exception('no checkpoint for {}'.format(checkpoint_path))

            # drop final dense layer and activation
            print("Modifying model and predicting representation")
            model.pop()
            model.pop()
            model.summary()

            # predict representation
            print("Predicting")
            X_rep = model.predict(X)
            del X
            gc.collect()

            if ensemble_visual:
                songs = np.unique(S)
                X_song = np.zeros((songs.shape[0], X_rep.shape[1]))
                Y_song = np.empty((songs.shape[0]), dtype="S10")
                for i, song in tqdm(enumerate(songs), total=len(songs), desc="Processing songs"):
                    xs = X_rep[S == song]
                    Y_song[i] = Y_original[S == song][0]
                    X_song[i, :] = np.mean(xs, axis=0)

                Y_song = np.array([song.decode('utf-8') for song in Y_song])

                X_rep = X_song
                Y_original = Y_song

            # fit tsne
            print("Fitting TSNE {}".format(X_rep.shape))
            tsne_model = TSNE()
            X_2d = tsne_model.fit_transform(X_rep)

            # save results
            save_path_CSV = os.path.join(cfg.save_path, 'CSV')
            save_path_PDF = os.path.join(cfg.save_path, 'PDF')
            save_path_PNG = os.path.join(cfg.save_path, 'PNG')

            os.makedirs(cfg.save_path, exist_ok=True)
            os.makedirs(save_path_CSV, exist_ok=True)
            os.makedirs(save_path_PDF, exist_ok=True)
            os.makedirs(save_path_PNG, exist_ok=True)
            print("Saving results")

            save_path_CSV = os.path.join(cfg.save_path, 'CSV', str(nb_classes) +
                                         '_' + str(slice_length) + '_' + str(random_state))
            save_path_PDF = os.path.join(cfg.save_path, 'PDF', str(nb_classes) +
                                         '_' + str(slice_length) + '_' + str(random_state))
            save_path_PNG = os.path.join(cfg.save_path, 'PNG', str(nb_classes) +
                                         '_' + str(slice_length) + '_' + str(random_state))

            if ensemble_visual:
                save_path_CSV += '_ensemble'
                save_path_PDF += '_ensemble'
                save_path_PNG += '_ensemble'

            # Save csv
            pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1],
                          'label': Y_original}).to_csv(
                save_path_CSV + '.csv', index=False)

            # Save figures (PNG, PDF)
            num_unique_y = len(np.unique(Y_original))
            sns.set_palette("Paired", n_colors=num_unique_y)

            plt.figure(figsize=(20, 20))
            sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=Y_original,
                            palette=sns.color_palette(n_colors=num_unique_y))

            plt.savefig(save_path_PNG + '.png')
            plt.savefig(save_path_PDF + '.pdf')

            plt.close()

            del Y, S, X_rep, X_2d, Y_original
            gc.collect()
