# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:37:44 2023

@author: marinamu
"""

import random
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import AUC, CategoricalAccuracy
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.utils import class_weight

from models import Models

random.seed(10)

class RunModel:

    def __init__(self, config, data_train, data_valid={}, should_print=True):
        self.config = config
        self.X_train = data_train["X"] # (n_samples, n_features)
        self.y_train = data_train["y"] # (n_samples, n_classes)
        self.input_shape = data_train["X"][0].shape
        self.data_valid = data_valid
        self.should_print = should_print  # if print the messages or not
        self.run_type = self.config.get("run_type", "")
        self.not_NN_classifiers = ["log_reg", "linear_svm", "knn"]
        self.model = None
        
    # ---------------------------------------------------------------------------------------------                
    def print_if_needed(self, message):
        if self.should_print:
            print(message)
            
    # ---------------------------------------------------------------------------------------------                
    def execute(self):
        # Compile the model and then train it:
        self.compile_model()
        history = self.fit_model()
        return history
    
    # ---------------------------------------------------------------------------------------------                
    def compile_model(self):
        if self.config["model_arc"] not in self.not_NN_classifiers:
            # Define model architecture:
            mdl_arc = Models(params=self.config, input_shape=self.input_shape)
            self.model = mdl_arc.model
            
            ### Define learning rate:
            # Calculate the number of training steps per epoch
            steps_per_epoch = len(self.X_train) // int(self.config["batch_size"])
            # Set decay_steps to be a multiple of steps_per_epoch (e.g., 5 times steps_per_epoch)
            decay_steps = self.config["num_steps_epoch"] * steps_per_epoch  # decay every 5 epochs
            
            lr_scheduler = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config["learning_rate"],
                decay_steps=decay_steps,
                decay_rate=self.config["decay_rate"])

            # Define optimizer:
            if self.config["optimizer"].lower() == "adam":
                optimizer = Adam(learning_rate=lr_scheduler)
            elif self.config["optimizer"].lower() == "rmsprop":
                optimizer = RMSprop(learning_rate=lr_scheduler)
            elif self.config["optimizer"].lower() == "sgd":
                optimizer = SGD(learning_rate=lr_scheduler)
            else:
                raise AssertionError("Mission aborted: no such optimizer: " + self.config["optimizer"])

            # Compile architecture:
            self.model.compile(loss=[self.config["loss"]],
                               optimizer=optimizer,
                               metrics=[self.config["metrics"], AUC(), CategoricalAccuracy()])

            self.print_if_needed('Done model compilation')

    # ==============================================================================================
    def fit_model(self, verbose=2, monitor='val_loss'):
        weights_dict = None
        history = None

        if len(self.y_train.shape)>1:
            y_train_class_idx = np.squeeze(np.argmax(self.y_train, axis=-1))
        else:
            y_train_class_idx = self.y_train.copy()          
            
        if self.config["model_arc"] not in self.not_NN_classifiers:
            if self.config["use_weights_balance"]:
                classes = np.unique(y_train_class_idx.tolist())
                # Calculate weights:
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y_train_class_idx.tolist())
                weights_dict = {i: weight for i, weight in
                                enumerate(class_weights)}  # keras requires a dictionary as an input for class_weight
                self.print_if_needed(f"Class weights are used, and they are: {weights_dict}")

            # Define early stopping:
            callbacks = []
            if self.config['early_stop'].get('evaluate', True):
                patience = self.config['early_stop']['patience']  # the number of epochs with no improvement
                if not self.data_valid:  # if empty dict
                    monitor = 'loss'
                early_stopping = EarlyStopping(monitor=monitor, patience=patience)
                callbacks = [early_stopping]
            
            print(f"#NaNs in X_train: {np.isnan(self.X_train).sum()}")
            print(f"#NaNs in y_train: {np.isnan(self.y_train).sum()}")
            if self.data_valid:  # if not empty dict
                history = self.model.fit(self.X_train, self.y_train,
                                         class_weight=weights_dict,
                                         epochs=self.config["epochs"],
                                         batch_size=self.config["batch_size"],
                                         validation_data=(self.data_valid["X"], self.data_valid["y"]),
                                         callbacks=callbacks, verbose=verbose)
            else:
                history = self.model.fit(self.X_train, self.y_train,
                                         class_weight=weights_dict,
                                         epochs=int(self.config["epochs"]),
                                         batch_size=int(self.config["batch_size"]),
                                         callbacks=callbacks, verbose=verbose)

        elif self.config["model_arc"] == "log_reg":
            self.model = LogisticRegression(max_iter=10000, penalty='l2', C=self.config["C"], verbose=verbose)
            self.print_if_needed('Done model compilation')
            self.print_if_needed("Training logistic regression model....")
            self.model.fit(self.X_train,y_train_class_idx)
            history = None

        elif self.config["model_arc"] == "linear_svm":
            self.model = svm.LinearSVC(C=self.config["C"], penalty=self.config["penalty"],
                                       verbose=1, multi_class='ovr')  # one-vs-the-rest     
            self.print_if_needed('Done model compilation')
            self.print_if_needed("Training SVM model....")
            self.model.fit(self.X_train, y_train_class_idx)
            history = None
            
        elif self.config["model_arc"] == "knn":
            from knn_classifier import KNNClassifier

            self.model = KNNClassifier(self.config)
            self.print_if_needed('Done model compilation')
            self.print_if_needed("Training KNN model....")
            self.model.fit(self.X_train, y_train_class_idx)
            history = None
            
        self.print_if_needed('Done model training')
        return history
