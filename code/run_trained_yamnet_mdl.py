# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:31:41 2023

@author: marinamu

Internally, the model extracts "frames" from the audio signal and processes batches of these frames. 
This version of the model uses frames that are 0.96 second long and extracts one frame every 0.48 seconds.

The model accepts a 1-D float32 Tensor or NumPy array containing a waveform of arbitrary length, 
represented as single-channel (mono) 16 kHz samples in the range [-1.0, +1.0]. 

The model returns 3 outputs, including the class scores, embeddings,and the log mel spectrogram. 
"""

import tensorflow as tf
import tensorflow_hub as hub  # conda install -c conda-forge tensorflow-hub
import numpy as np
import csv
import scipy

import matplotlib.pyplot as plt
from IPython.display import Audio


class RunYAMNET:

    def __init__(self, waveform=None, original_fs=None, desired_fs=16000, plot_result=False, should_print=True):
        self.waveform = waveform
        self.original_fs = original_fs
        self.desired_fs = desired_fs
        self.plot_result = plot_result
        self.should_print = should_print

        self.model = None
        self.class_names = None

    # =============================================================================================
    def print_if_needed(self, message):
        if self.should_print:
            print(message)

    # =================================================================================================
    def run_all(self):
        self.load_model_labels()
        self.prepare_audio()
        self.print_if_needed("Predict classes")
        self.predict_class()
        self.get_out_class()

        if self.plot_result:
            self.plot_spec_class()

    # =================================================================================================
    def load_model_labels(self):
        self.print_if_needed("Load model")
        self.load_model()
        self.print_if_needed("Load model labels")
        self.load_labels()

    # =================================================================================================
    def prepare_audio(self):
        # self.print_if_needed("Ensure sample rate")
        self.waveform = self.ensure_sample_rate()
        self.norm_audio()

    # =================================================================================================
    def load_model(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        # =================================================================================================

    def load_labels(self):
        """
        The labels file will be loaded from the models assets.
        """

        def class_names_from_csv(class_map_csv_text):
            """Returns list of class names corresponding to score vector."""
            class_names = []
            with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    class_names.append(row['display_name'])
            return class_names

        class_map_path = self.model.class_map_path().numpy()
        self.class_names = class_names_from_csv(class_map_path)

    # =================================================================================================
    def ensure_sample_rate(self):
        """
        Add a method to verify and convert a loaded audio is on the proper sample_rate (16K), otherwise it would affect the model's results.
        """
        # Resample waveform if required.
        if self.original_fs != self.desired_fs:
            desired_length = np.int(round(np.float(len(self.waveform)) /
                                          self.original_fs * self.desired_fs))
            return scipy.signal.resample(self.waveform, desired_length)
        else:
            return self.waveform

    # =================================================================================================
    def norm_audio(self):
        """
        The `wav_data` needs to be normalized to values in `[-1.0, 1.0]` 
        (as stated in the model's [documentation](https://tfhub.dev/google/yamnet/1)).
        """
        if isinstance(self.waveform, int):
            self.print_if_needed("Normalize audio to values in [-1.0, 1.0]")
            self.waveform = self.waveform / tf.int16.max

    # =================================================================================================
    def predict_class(self):
        """
        Using the data already prepared, you just call the model and get the: 
            scores, embeddings, and the spectrogram.
        """

        # Run the model, check the output.
        self.scores, self.embeddings, self.spectrogram = self.model(self.waveform)

    # =================================================================================================
    def get_out_class(self):
        # Check the output.
        self.scores_np = self.scores.numpy()
        self.infered_class = self.class_names[self.scores_np.mean(axis=0).argmax()]
        # self.print_if_needed(f'The main class of the sound is: {self.infered_class}')
        return self.infered_class

    # =================================================================================================
    @staticmethod
    def listen_audio(wav_data, sample_rate):
        # Listening to the wav file.
        Audio(wav_data, rate=sample_rate)

    # =================================================================================================
    def plot_spec_class(self, top_n=15):
        """
        ## Visualization

        YAMNet also returns some additional information that we can use for visualization.
        Plot the Waveform, spectrogram, and the top classes inferred.
        """
        # Convert to numpy:
        scores_np = self.scores.numpy()
        spectrogram_np = self.spectrogram.numpy()

        # Plot:
        plt.figure(figsize=(10, 6))

        # Plot the waveform
        plt.subplot(3, 1, 1)
        plt.plot(self.waveform)
        plt.xlim([0, len(self.waveform)])

        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(self.scores, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        plt.subplot(3, 1, 3)
        plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

        # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding - 0.5, self.scores.shape[0] + patch_padding - 0.5])
        # Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [self.class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_n, 0]))
