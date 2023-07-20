# -*- coding: utf-8 -*-

import os
import gc
import pandas as pd
import numpy as np

from src.config import Config
from src.trainer import TrainingProcess


def save_results(scores, pooling_scores, summary_metrics_output_folder, slice_len):
    """
    Saves the scores and pooling scores data as CSV files in the given directory.

    Args:
    - scores (numpy.ndarray): Array of scores data.
    - pooling_scores (numpy.ndarray): Array of pooling scores data.
    - summary_metrics_output_folder (str): Path of the directory where the CSV files will be saved.
    - slice_len (int): Length of the slice.

    Returns: None
    """
    os.makedirs(summary_metrics_output_folder, exist_ok=True)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f'{summary_metrics_output_folder}/{slice_len}_score.csv')
    pooling_scores_df = pd.DataFrame(pooling_scores)
    pooling_scores_df.to_csv(
        f'{summary_metrics_output_folder}/{slice_len}_pooled_score.csv')


if __name__ == '__main__':
    cfg = Config()
    scores = []
    pooling_scores = []
    # slice_len = 157
    verboso = False

    for slice_len in cfg.slice_lengths:
        for random_state in cfg.random_state_list:
            trainer = TrainingProcess(
                song_folder=cfg.song_folder,
                nb_classes=cfg.nb_classes,
                slice_length=slice_len,
                learning_rate=cfg.learning_rate,
                plots=cfg.plots,
                load_checkpoint=cfg.load_checkpoint,
                save_metrics=cfg.save_metrics,
                save_metrics_folder=cfg.save_metrics_folder,
                save_weights_folder=cfg.save_weights_folder,
                batch_size=cfg.batch_size,
                nb_epochs=cfg.nb_epochs,
                album_split=cfg.album_split,
                random_states=random_state)

            # Load data
            print("Loading dataset...")
            data_dict = trainer.load_data()
            gc.collect
            if verboso:
                # dict_keys(['Y_train', 'X_train', 'S_train', 'Y_test', 'X_test', 'S_test', 'Y_val', 'X_val', 'S_val'])
                #   Y_train: <class 'list'>, length=926, in_type=<class 'str'>
                #   X_train: <class 'list'>, length=926, in_type=<class 'numpy.ndarray'>
                #   S_train: <class 'list'>, length=926, in_type=<class 'str'>
                #   Y_test:  <class 'list'>, length=254, in_type=<class 'str'>
                #   X_test:  <class 'list'>, length=254, in_type=<class 'numpy.ndarray'>
                #   S_test:  <class 'list'>, length=254, in_type=<class 'str'>
                #   Y_val:   <class 'list'>, length=233, in_type=<class 'str'>
                #   X_val:   <class 'list'>, length=233, in_type=<class 'numpy.ndarray'>
                #   S_val:   <class 'list'>, length=233, in_type=<class 'str'>
                for key in data_dict.keys():
                    print(f"{key}: {type(data_dict[key])}, "
                          f"length={len(data_dict[key])}, "
                          f"in_type={type(data_dict[key][0])}")
                print()

            # Slice data
            print("Slicing songs...")
            sliced_dict = trainer.slice_data(data_dict)
            del data_dict
            gc.collect
            if verboso:
                # sliced_dict(['Y_train', 'X_train', 'S_train', 'Y_test', 'X_test', 'S_test', 'Y_val', 'X_val', 'S_val'])
                #   X_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>
                #   Y_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>
                #   S_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>
                #   X_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>
                #   Y_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>
                #   S_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>
                #   X_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>
                #   Y_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
                #   S_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
                for key in sliced_dict.keys():
                    print(f"{key}: {type(sliced_dict[key])}, "
                          f"length={len(sliced_dict[key])}, "
                          f"in_type={type(sliced_dict[key][0])}")
                print()

            # Print label counts
            if verboso:
                artist, counts = np.unique(
                    sliced_dict['Y_train'], return_counts=True)
                rows = ["{:<30} {:<10}".format(band, count)
                        for band, count in zip(artist, counts)]
                print(f"Training set label counts ({type(artist)})")
                print("\n".join(rows))
                print()

            # Encode data
            encoded_data = trainer.encode_and_reshape_data(sliced_dict)
            del sliced_dict
            gc.collect
            if verboso:
                print("encoded_data: tuple[dict, LabelEncoder, OneHotEncoder]")
                # Y_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>
                # X_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.ndarray'>
                # S_train: <class 'numpy.ndarray'>, length=26433, in_type=<class 'numpy.str_'>
                # Y_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>
                # X_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.ndarray'>
                # S_val:   <class 'numpy.ndarray'>, length=2857,  in_type=<class 'numpy.str_'>
                # Y_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>
                # X_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.ndarray'>
                # S_test:  <class 'numpy.ndarray'>, length=3367,  in_type=<class 'numpy.str_'>
                for key in encoded_data[0].keys():
                    print(f"{key}: {type(encoded_data[0][key])}, "
                          f"length={len(encoded_data[0][key])}, "
                          f"in_type={type(encoded_data[0][key][0])}")
                print()

            # Build model
            model = trainer.build_model(encoded_data[0], verboso)
            gc.collect

            # Load weights and customize model
            ModelCheckpoint, EarlyStopping = trainer.load_weights(model)
            gc.collect

            # Train the model
            if cfg.train:
                trainer.train_model(
                    model, ModelCheckpoint, EarlyStopping, encoded_data[0])
            gc.collect

            # Score the model
            score, pooling_score = trainer.score_model(
                model=model, data_dict=encoded_data[0], le=encoded_data[1], ext="pdf")
            gc.collect

            # Save results
            scores.append(score["weighted avg"])
            pooling_scores.append(pooling_score['weighted avg'])
            save_results(scores, pooling_scores,
                         cfg.summary_metrics_output_folder, slice_len)
            gc.collect
