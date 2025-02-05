# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:14:24 2023

@author: marinamu
"""

import os
import numpy as np
import itertools
import pandas as pd
import random

from sklearn.model_selection import StratifiedGroupKFold

from evaluate_model import EvaluateModel
from embeddings_augmentation import AugmentEmbeddings
from common import keep_N_random_events
from run_model import RunModel

random.seed(1337)


class MyParamsSearch:
    def __init__(self, config, data_train, search_params, save_path, n_folds=4):
        self.config = config
        self.data_train = data_train
        self.search_params = search_params
        self.n_folds = n_folds
        self.save_path = save_path

        self.csv_filename_path = f"{self.save_path}/Mean_CV_results.csv"
        self.mean_cv_all_results = []  # mean results of each combination
        self.best_params_ = {}  # the combination of parameters deriving the best performance
        self.df_results = pd.DataFrame()
        self.idx_comb = 1  # The combination to start tuning from:
        # Create the possible combinations:
        self.combinations = self.get_possible_combinations()
        self.var_by_split = "group"

    # ----------------------------------------------------------------------------------------------
    def get_possible_combinations(self):
        # 1: Create the possible combinations:
        keys = self.search_params.keys()
        values = (self.search_params[key] for key in keys)
        combinations = [dict(zip(keys, comb)) for comb in itertools.product(*values)]
        # 2. Shuffle the combinations:
        random.shuffle(combinations)  # shuffle the 'combinations' variable
        if 'random' in self.search_params["search_type"]:
            # 3. Check only n_combs:
            combinations = combinations[0:min(
                [self.search_params['n_combs'][0], len(combinations)])]  # choose first "n_combs"s to run
        return combinations

    @staticmethod
    def get_specific_indices(data, indices):
        if isinstance(data, list):
            # data is a list of ndarrays
            result = [data[i] for i in indices]
        elif isinstance(data, np.ndarray):
            # data is a single ndarray
            result = data[indices]
        else:
            raise TypeError("data must be a list of ndarrays or a single ndarray")
        return result

    # ----------------------------------------------------------------------------------------------
    def run_search_params(self, scoring='cohen_kappa', ascending=False):  # the higher = the best
        
        all_cv_results = []

        # For each combination run cross-validation and calculate mean performance:
        for i_comb, comb in enumerate(self.combinations):
            if i_comb < (self.idx_comb - 1):  # skip the done combinations
                continue

            # Initialize a list storing the performance of each fold:
            cv_results = []  # results of all cross validation splits for one specific combination of parameters:

            # Iterate over each fold
            if len(self.data_train["y"].shape) > 1:
                y = np.argmax(self.data_train["y"], axis=-1)
            else:
                y = self.data_train["y"].copy()

            # Define the k-fold splitter:
            cv = StratifiedGroupKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
            print(f'\nNow Tuning {i_comb + 1}/{len(self.combinations)}: {comb}')
            for i, (train_idx, valid_idx) in enumerate(
                    cv.split(self.data_train["X"], y, self.data_train[self.var_by_split])):

                data_train_cv, data_valid_cv = {}, {}

                for key in self.data_train.keys():
                    data_train_cv[key] = self.get_specific_indices(self.data_train[key], train_idx)
                    data_valid_cv[key] = self.get_specific_indices(self.data_train[key], valid_idx)
                
                # Keep only N first events in Train data:
                data_train_reduced = keep_N_random_events(
                    data_train_cv,
                    n_p_keep=self.config["train_prec_take"],
                    class_label=self.config["mapping_class"]["Speech"])
                
                # Apply augmentation to train dataset only:
                if self.config["if_augment"]:
                    print(f'Apply {self.config["augment_type"]} augmentation to train embeddings')
                    augmentor = AugmentEmbeddings(self.config, data_train_reduced)
                    x_resampled, y_resampled = augmentor.X_resampled, augmentor.y_resampled
                    data_train = {'X': x_resampled, 'y': y_resampled}
                else:
                    data_train = data_train_reduced.copy()

                # Define/compile the Classifier:
                combined_params = {**self.config, **comb}
                print("params_search_mine - RunModel:")
                mdl_cls = RunModel(config=combined_params, data_train=data_train)
                mdl_cls.execute()
                # Evaluate on validation dataset:
                print("params_search_mine - EvaluateModel:")
                eval_cls = EvaluateModel(mdl_cls.model, data_valid_cv,
                                         class_names=list(set(self.config["mapping_class"].values())))
                eval_cls.model_extractor = self.config["embeddings_config"].get("model_extractor", "")
                eval_cls.run_type = self.config.get("run_type", "translearn")
                eval_cls.execute()
                results_cv = eval_cls.results  # dict
                cv_results.append(results_cv)  # list of dicts
                all_cv_results.append(results_cv.update(comb))
                # K.clear_session()
                print(
                    f"Done comb={i_comb + 1}/{len(self.combinations)}: fold {i + 1}/{self.n_folds} of Train-Validation in parameters tuning")

            # Calculate mean results of the k-folds cross validation of a specific combination of parameters:
            mean_results_comb = dict(comb)  # create a copy of comb dict
            # Add a series of mean values per column to the dict of comb parameters
            mean_results_comb.update(pd.DataFrame(cv_results).mean(numeric_only=True).to_dict())
            self.mean_cv_all_results.append(mean_results_comb)  # list of dicts
            # Save the combination results to csv file:
            print("mean_results_comb: {}".format(mean_results_comb))
            self.save_mean_cv_to_csv(pd.DataFrame.from_dict(mean_results_comb, orient='index').transpose())

        # Convert list of dicts to dataframe:
        self.df_results = pd.DataFrame(self.mean_cv_all_results)

        # Sort the table: from best to worse:
        self.df_results.sort_values([scoring], axis=0, ascending=[ascending], inplace=True)
        # Get the parameters that derive best performance:
        self.best_params_ = self.get_best_params()

    # --------------------------------------------------------------------------------------------
    def get_best_params(self):
        # Get the highest score
        best_params = self.df_results.iloc[0].to_dict()
        print("\nBest parameters: {}".format(best_params))
        return best_params

    # ---------------------------------------------------------------------------------------------
    def save_mean_cv_to_csv(self, df_results: pd.DataFrame):
        """ Save the mean results of the cross validation of one set of parameters """
        # Create the save_path_fold folder if doesnt exist already:
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        # Save the mean results to CSV file:
        if os.path.isfile(self.csv_filename_path):  # if the files already exists then append at the end:
            df_results.to_csv(self.csv_filename_path, mode='a', index=False, header=False)
        else:  # if the files doesn't exist then create the file and write into it:
            df_results.to_csv(self.csv_filename_path, mode='w', index=False)

    # ---------------------------------------------------------------------------------------------
    def load_results(self, scoring='cohen_kappa', ascending=False):
        """ Load the mean results of the cross validation of all parameters combinations """
        # Load the excel file with the tuning results:
        self.df_results = pd.read_csv(self.csv_filename_path)
        # Sort the table: from best to worse:
        self.df_results.sort_values([scoring], axis=0, ascending=[ascending], inplace=True)
