# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:48:12 2022

@author: marinamu
"""
from pathlib import Path
import os
from datetime import datetime
import pandas as pd
import numpy as np
import sys

from common import get_database_path

class CreateCodebook:
    def __init__(self, config, codebook_filename='CodeBook'):
        self.config = config  # dict
        self.codebook_filename = codebook_filename

        self.config_CB_name = None  # configuration name
        self.CB_file_save_path = None  # the path of the Codebook file
        self.config_dir_save_path = None  # the path where the embeddings of this configuration are saved
        self.date_now = datetime.today().strftime('%d%m%Y_%H%M%S')

    # Execute all -> get the current configuration CB name
    # ---------------------------------------------------------------------------------------------      
    def execute(self):
        # 1. get the path where the CB is saved:
        self.get_CB_file_save_path()
        # 2. Get configuration CB name:
        self.get_config_CB_name()
        print("The configuration code name is: {}".format(self.config_CB_name))

    # Get the path where the file of description of Codebooks is saved:
    # --------------------------------------------------------------------------------------------- 
    def get_CB_file_save_path(self):
        # Get the path of the current folder (where the script is):
        # cur_dir = Path(__file__).parent.absolute()
        self.CB_file_save_path = get_database_path('codebooks') / Path(
            f"{self.codebook_filename}.txt")

    # Get the configuration CB name:
    # --------------------------------------------------------------------------------------------- 
    def get_config_CB_name(self):
        config_df = pd.DataFrame.from_dict([self.config])  # convert to dataframe
        config_df.fillna(value=np.nan, inplace=True)  # Convert None with nan. 05.09.22

        # Check if the CB file already exists:
        if not os.path.isfile(self.CB_file_save_path):  # doesnt exist -> create
            print('The CB file does not exist')
            self.config_CB_name = "config_{}".format(self.date_now)
            # Insert the name of the configuration as the first column in df:
            config_df.insert(loc=0, column='config_CB_name', value=self.config_CB_name)
            # Save the configuration to file:
            config_df.to_csv(self.CB_file_save_path, mode='w', index=False, header=True, sep='\t')

        else:  # if the CB file already exists
            # Read the CB file:
            # CB_df = pd.read_csv(self.CB_file_save_path, sep='\t', index_col=False)  # CB dataframe
            CB_df = pd.read_csv(self.CB_file_save_path, sep='\t', index_col=False)  # CB dataframe
            # Convert True to 1 and False to 0
            CB_df = CB_df.applymap(lambda x: 1. if x == "True" else 0. if x == "False" else x) # 18.07.24
            
            # Check if all keys in the current configuration are in CB:
            if all(key in list(CB_df.columns) for key in list(config_df.columns)):
                ### Find the row of the current configuration in the CB:
                # Convert all empty cells in CB_df to None string:
                tmp_CB_df = CB_df.applymap(lambda x: str(x) if x is not None else None)
                # Take the CB_df columns:
                all_columns = set(tmp_CB_df.columns)
                # To not change the original configuration file, copy to new variable:
                filter_dict = self.config.copy()
                # Convert True to 1 and False to 0
                # filter_dict = {k: (1. if v == True else 0. if v == False else v) for k, v in filter_dict.items()} # 18.07.24
                # Find missing columns in config that are in CB_df:
                missing_columns = all_columns - set(self.config.keys())
                # Set None for these columns and add them to the config dict:
                filter_dict.update({column: None for column in missing_columns})
                # Remove the key of the config name (empty in config):
                filter_dict.pop('config_CB_name')
                # Convert all empty cells in config to None string:
                filter_dict = {key: str(value) if value is not None else 'nan' for key, value in
                               filter_dict.items()}
                # Prepare the string for query search:
                query_string = ' and '.join(
                    [f"{key} == '{value}'" if isinstance(value, str) else f"{key} == {value}" for
                     key, value in filter_dict.items()])
                # Find the row in CB_df of the parameters in config:
                row_in_CB = tmp_CB_df.query(query_string).reset_index(drop=True)
                # if the current configuration is found in the CB:
                if row_in_CB.shape[0] > 0:
                    if row_in_CB.shape[0] == 1:
                        # Convert numeric variables back to numeric:
                        for col in row_in_CB.columns:
                            try:
                                row_in_CB[col] = int(row_in_CB[col])
                            except (ValueError, TypeError):
                                try:
                                    row_in_CB[col] = float(row_in_CB[col])
                                except (ValueError, TypeError):
                                    pass
                        print('The CB file exists AND the specific configuration is in the CB')
                        self.config_CB_name = row_in_CB['config_CB_name'].values[0]
                        return
                    else:
                        print(f'The CB file exists AND there are multiple rows in CB that fit current configuration: {row_in_CB}')
                        sys.exit()
                else:  # doesnt exist in CB
                    flag = 0
            else:  # not all keys of the current configuration exist in CB
                print('Not all keys of the current configuration exist in CB')
                flag = 0

            # The current configuration is not in the CB:
            if flag == 0:  # does not exist in CB -> create a new config_CB_name and append to CB table:
                print('The Codebook file exists but the specific configuration does not')
                # Create the current configuration CB name:
                self.config_CB_name = "config_{}".format(self.date_now)
                config_df.insert(loc=0, column='config_CB_name', value=self.config_CB_name)
                # Append the current configuration to existing configurations table:
                df_united = pd.concat([CB_df, config_df], axis=0, ignore_index=True)
                # Save the updated Codebook to file:
                df_united.to_csv(self.CB_file_save_path, mode='w', index=False, header=True,sep='\t')
