# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:03:21 2023

@author: marinamu
"""

# import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class PrepareData_Model():
    '''
    A class for preparing data by converting categorical labels to numeric values and creating dummy variables.

    Parameters:
        mapping (dict): A dictionary mapping categorical labels to numeric values.

    Methods:
        __init__(self, mapping):
            Initializes an instance of the PrepareData_Model class.

        execute(self, X_dict, y_dict):
            Executes the data preparation process by converting input data.

        convert_X_arr(self, X_dict):
            Converts a dictionary of feature arrays into a 2D array.

        convert_y_arr(self, y_dict):
            Converts a dictionary of label arrays into a string array.

        convert_y_dummy(self, y_string_array):
            Converts a string array of labels to dummy variables.
    '''

    def __init__(self, mapping, convert2dummy):
        self.mapping = mapping
        self.convert2dummy = convert2dummy

    # ==============================================================================================
    def execute(self, data):
        """
        Executes the data preparation process by converting input data.

        Parameters:
            data (dict): A dictionary containig X and y: 
                X (dict): A dictionary of feature arrays.
                y (dict): A dictionary of label arrays.

        Returns:
            X (ndarray): The converted feature data as a 2D array. (n_samples, n_features)
            y (ndarray): The converted label data as a 2D array of dummy variables. (n_samples, 3)
        """

        X = self.convert_X_arr(data["X"])  # (n_samples, n_features)
        y_string_array = self.convert_y_arr(data["y"])
        # Convert to one-hot: column for each class
        y = self.convert_string2arr(y_string_array)  # (n_samples, 3)
        group = self.convert_y_arr(data["group"])  # (n_samples, )
        return {"X": X, "y": y, 'group': group}

    # ==============================================================================================
    def convert_X_arr(self, X_dict):  # 3D array: n_samples x n_features
        """
        Converts a dictionary of feature arrays into a 2D array.

        Parameters:
            X_dict (dict): A dictionary of feature arrays.

        Returns:
            ndarray: The feature data as a 2D array.
        """
        X_2d = np.array([sub_v for k, v in X_dict.items() for sub_v in v])
        if X_2d.ndim == 3:  # (n_samples, n_features, 1)
            return np.squeeze(X_2d)  # (n_samples, n_features)
        else:
            return X_2d  # (n_samples, n_features)

    # ==============================================================================================
    def convert_y_arr(self, y_dict):  # string array
        """
        Converts a dictionary of label arrays into a string array.

        Parameters:
            y_dict (dict): A dictionary of label arrays.

        Returns:
            ndarray: The label data as a string array.
        """
        return np.array([sub_v for k, v in y_dict.items() for sub_v in v])

    # ==============================================================================================
    def convert_string2arr(self, y_string_array):  # n_samples x num_classes
        """
        Converts a string array of labels to dummy variables.

        Parameters:
            y_string_array (ndarray): A string array of labels.

        Returns:
            ndarray: The label data as a 2D array of dummy variables.
        """
        # Map the string array to numeric values using the mapping settings
        numeric_array = [self.mapping[value] for value in y_string_array]

        if self.convert2dummy:
            
            # Create dummy variables
            # return pd.get_dummies(df['Class']).values # one-hot array: column for each class
            label_encoder = LabelEncoder()
            # Fit label encoder and transform class names to integer labels
            encoded_labels = label_encoder.fit_transform(numeric_array)
            # Convert the integer labels to one-hot encoding
            num_classes = len(label_encoder.classes_)
            return to_categorical(encoded_labels, num_classes=num_classes)

        else:
            return np.asarray(numeric_array)  # array of number of classes: each sample has its class
