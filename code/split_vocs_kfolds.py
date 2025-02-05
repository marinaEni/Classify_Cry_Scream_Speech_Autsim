# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:04:26 2021

@author: marinamu
"""
from sklearn.model_selection import StratifiedGroupKFold  # Takes group information into account to avoid building folds with imbalanced class distributions


class SplitVocsKFolds:

    def __init__(self, config, data_df, should_print=True):
        self.data_df = data_df
        self.k_folds = config.get('k_fold', 5)  # default: 5 folds
        self.should_print = should_print

        self.var_by_split = "group"

    def print_if_needed(self, message):
        if self.should_print:
            print(message)

    # =============================================================================================
    def execute(self):
        '''
        First, create one long dataframe where each row is information of an event, from all recordings.
        Second, split the whole data, such as all audio events of the same recording will be in the same dataset.
        This will return a datframe for each dataset, that includes the information of all events off all recordings in the dataset.
        '''
        # Create train-test folds:
        self.train_data, self.test_data = self.create_train_test(df=self.data_df,
                                                                 k_folds=self.k_folds)
        return self.train_data, self.test_data

    # =============================================================================================
    def create_train_test(self, df, k_folds):
        """
        Creates train and test datasets using stratified group k-fold cross-validation.
    
        Args:
            df (DataFrame): The input DataFrame containing the data.
            k_folds (int): The number of folds for cross-validation.
            var_by_split (string): The variably to split the data by (child name / recording name)

        Returns:
            train_data (dict): A dictionary containing the train datasets for each fold.
            test_data (dict): A dictionary containing the test datasets for each fold.
        """
        # Initialize stratified folds with non-overlapping groups/recordings:
        gkf = StratifiedGroupKFold(n_splits=k_folds, random_state=42, shuffle=True)

        # Initialize dictionaries to store train and test datasets
        train_data, test_data = dict(), dict()

        # Iterate over each fold
        for i, (train_idx, test_idx) in enumerate(gkf.split(X=df, y=df["event"], groups=df[self.var_by_split])):
            # Split the data into train and test sets using grouped stratified sampling
            # train_idx, test_idx = next(gkf.split(X=df, y=df["event"], groups=df["Recording"]))

            train_data[f"fold{i}"] = df.iloc[train_idx].reset_index(drop=True)
            test_data[f"fold{i}"] = df.iloc[test_idx].reset_index(drop=True)

            # Check if there no common recordings between datasets:
            assert ~sum(train_data[f"fold{i}"][self.var_by_split].isin(test_data[f"fold{i}"][self.var_by_split])), \
                'ERROR: common recordings train and test'

            # If runs to here than no error.
            # Print information about the train and test datasets for the current fold
            self.print_if_needed("NO common recordings between train-test")
            self.print_if_needed(
                "\nNumber of events for X_train:\n{}".format(train_data[f"fold{i}"]["event"].value_counts()))
            self.print_if_needed(
                "Number of events for X_test:\n{}".format(test_data[f"fold{i}"]["event"].value_counts()))
            self.print_if_needed("\nX_train shape: {} events".format(train_data[f"fold{i}"].shape))
            self.print_if_needed("X_test shape: {} events".format(test_data[f"fold{i}"].shape))

        return train_data, test_data
