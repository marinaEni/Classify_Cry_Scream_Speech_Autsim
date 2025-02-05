# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:03:40 2023

@author: marinamu
"""
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import numpy as np


class AugmentEmbeddings:
    def __init__(self, config, data):
        self.augment_type = config["augment_type"]
        self.data = data # dict: X = (n_samples, n_embeddings) and y=(n_samples, n_classes)
        self.X_resampled, self.y_resampled = self.apply_augmentation()
    
    # ---------------------------------------------------------------------------------------------
    def apply_augmentation(self):
        if self.augment_type == 'SMOTE':
            
            # Initialize SMOTE algorithm:
            smote = SMOTE(random_state=42, sampling_strategy="minority")
            
            # In case of one-hot convert y_resampled back to array (n_samples,):
            if self.is_one_hot(self.data["y"]): 
                # Reshape y to (n_samples,)
                y_reshaped = self.data["y"].argmax(axis=-1)  # (n_samples, )
                # Make the augmentation:
                X_resampled, y_resampled = smote.fit_resample(self.data["X"], y_reshaped)
                # Convert back to one-hot:
                y_resampled_1hot = self.one_hot_encode(y_resampled, 
                                                       num_classes=self.data["y"].shape[-1])
                print("#Samples before Augmentation: {}, and after: {}".format(
                    self.data["X"].shape[0], X_resampled.shape[0]))
                print("Number of samples for each class before Augmentation: {}, and after: {}". format(
                    self.count_samples(y_reshaped), self.count_samples(y_resampled)))
                
                return X_resampled, y_resampled_1hot
            
            else: # return resampled y as array (resampled n_samples,)
                X_resampled, y_resampled = smote.fit_resample(self.data["X"],
                                                              self.data["y"])
                print("#Samples before Augmentation: {}, and after: {}".format(
                    self.data["X"].shape[0], X_resampled.shape[0]))
                print("Number of samples for each class before Augmentation: {}, and after: {}". format(
                    self.count_samples(self.data["y"]), self.count_samples(y_resampled)))
                
                return X_resampled, y_resampled
            
        elif 'BorderSMOTE' in self.augment_type:
            border_method = 'borderline-1' if self.augment_type=="BorderSMOTE1" else 'borderline-2'
            # Initialize SMOTE algorithm:
            smote = BorderlineSMOTE(random_state=42, sampling_strategy="minority", kind=border_method)
            
            # In case of one-hot convert y_resampled back to array (n_samples,):
            if self.is_one_hot(self.data["y"]): 
                # Reshape y to (n_samples,)
                y_reshaped = self.data["y"].argmax(axis=-1)  # (n_samples, )
                # Make the augmentation:
                X_resampled, y_resampled = smote.fit_resample(self.data["X"], y_reshaped)
                # Convert back to one-hot:
                y_resampled_1hot = self.one_hot_encode(y_resampled, 
                                                       num_classes=self.data["y"].shape[-1])
                print("#Samples before Augmentation: {}, and after: {}".format(
                    self.data["X"].shape[0], X_resampled.shape[0]))
                print("Number of samples for each class before Augmentation: {}, and after: {}". format(
                    self.count_samples(y_reshaped), self.count_samples(y_resampled)))
                
                return X_resampled, y_resampled_1hot
            
            else: # return resampled y as array (resampled n_samples,)
                X_resampled, y_resampled = smote.fit_resample(self.data["X"],
                                                              self.data["y"])
                print("#Samples before Augmentation: {}, and after: {}".format(
                    self.data["X"].shape[0], X_resampled.shape[0]))
                print("Number of samples for each class before Augmentation: {}, and after: {}". format(
                    self.count_samples(self.data["y"]), self.count_samples(y_resampled)))
                
                return X_resampled, y_resampled
        else:
            raise AssertionError("No such augmentation method: " + self.augment_type)
    
    # ---------------------------------------------------------------------------------------------
    # Check if array is one-hot:
    def is_one_hot(self, array):
        if array.ndim != 2:
            return False
    
        # Check if each row has only one non-zero value
        num_nonzero = np.sum(array != 0, axis=1)
        if np.all(num_nonzero == 1):
            return True
        else:
            return False
    
    # ---------------------------------------------------------------------------------------------
    # Convert array of integers to one-hot encoding:
    def one_hot_encode(self, array, num_classes):
        # Create an identity matrix of shape (num_classes, num_classes)
        identity_matrix = np.eye(num_classes)
    
        # Use the array of integers as indices to select rows from the identity matrix.
        one_hot_encoded = identity_matrix[array]
    
        return one_hot_encoded
    
    # ---------------------------------------------------------------------------------------------
    # Count numbe rof samples for each class:
    def count_samples(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        return class_counts