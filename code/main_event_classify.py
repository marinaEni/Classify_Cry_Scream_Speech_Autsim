# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:59:40 2021

@author: marinamu
"""

"""
Extract embeddings using pre-trained model and fine-tune using downstream model on Soroka database.
"""

from pathlib import Path
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import numpy as np
import pandas as pd
from scipy import stats
import argparse

from datetime import date, datetime
import csv
import yaml
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import model_from_json
import soundfile as sf
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from tensorflow.keras import backend as K

random.seed(10)

mainpath = Path(__file__).parent.absolute()
os.chdir(mainpath)

from split_vocs_kfolds import SplitVocsKFolds
from generate_load_vocs import GenerateLoadVocs
from extract_embeddings_dataset import ExtractEmbeddings
from run_model import RunModel
from prepare_data_model import PrepareData_Model
from evaluate_model import EvaluateModel
from params_search_mine import MyParamsSearch
from common import get_database_path, keep_N_random_events, load_yaml
from embeddings_augmentation import AugmentEmbeddings
from tic_toc_class import tic_toc
from create_codebook import CreateCodebook
from load_manual_segments import LoanManualSegments

DPI = 300
FONT = {'family': 'Times New Roman', 'size': 16}


# extract configuration properties:
# =================================================================================================    
def extract_config():
    parser = argparse.ArgumentParser(description="Run event classification with external data")
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file')

    # Parse the arguments
    options = parser.parse_args()

    # Check if the config file path is provided
    if not options.config:
        print("Error: The configuration file path is not specified.")
    else:
        try:
            config_dict = load_yaml(file_pointer=options.config)
            print("Configuration successfully loaded")
            # Proceed with the rest of your code using config_dict
        except Exception as e:
            print(f"Failed to load configuration: {e}")

    # Load configurations:
    main_config = config_dict.get('main_config', dict())

    return options, main_config


# Extract/load vocalizations for each recoridng and store in a dict (vocs per rec):
# =================================================================================================
def extract_vocalizations(config):
    print("Extract/load vocalizations")
    vs_spec_cl = GenerateLoadVocs(config=config["vocal_segment_config"],
                                  recs_names_list=config["rec_list_yaml"])
    vs_spec_cl.execute_config()
    return vs_spec_cl.vocal_segments


# Load manual segments for each recording and store in a dict (df per rec):
# =================================================================================================
def extract_manual_segments(config):
    print("Load manual segmentations")
    manual_segs_cl = LoanManualSegments(config=config["vocal_segment_config"],
                                        recs_names_list=config["rec_list_yaml"])
    manual_segs_cl.execute_config()
    return manual_segs_cl.manual_segments


# Convert dict of dataframes (df for each recording) to one long dataframe and add a column of recording id
# =================================================================================================
def vocal_segments_to_df(vocs_per_rec_dict):
    dfs = []  # List to store DataFrames with "Recording" column added

    for i, (rec, df) in enumerate(vocs_per_rec_dict.items()):
        df["Recording"] = rec
        df["group"] = rec  # Assuming you also want to add a "group" column
        dfs.append(df)

    print("Convert dict of dataframes to long dataframe with Recording column")
    # Concatenate all recordings' vocal segments into one long dataframe:
    vocs_per_rec_dict_df = pd.concat(dfs, ignore_index=True)
    # if vocs_per_rec_dict_df doesnt include all the recs that in vocs_per_rec_dict, then maybe they have no vocs
    return vocs_per_rec_dict_df


# Replace Echolalia with Speech
# =================================================================================================
def replace_echo_speech(vocs_per_rec_dict):
    # Iterate on vocalizations dataframe of each recording:
    for rec, df in vocs_per_rec_dict.items():
        # Replace 'Echolalia' with 'Speech':
        df['event'] = df['event'].replace(to_replace=['Echolalia'], value=['Speech'])

    return vocs_per_rec_dict

# # =================================================================================================
def split_data_train_test_ktimes(config, vocs_df):
    """
    Splits the data into train and test sets using the SplitVocsKFolds class k times.

    Args:
        config (object): The configuration object.
        vocs_df (DataFrame): The input DataFrame containing the vocal segments data.

    Returns:
        train_data_dict (dict): A dictionary containing the train datasets for each fold.
        test_data_dict (dict): A dictionary containing the test datasets for each fold.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
    """
    k_folds = config["k_fold"]
    print(f"Split data to train-test {k_folds} folds")
    # Create an instance of the SplitVocsKFolds class
    spl_cls = SplitVocsKFolds(config, vocs_df)
    # Execute the splitting process and obtain the train and test datasets as dictionaries
    train_data_dict, test_data_dict = spl_cls.execute()
    return train_data_dict, test_data_dict


# Extract embeddings for each dataset and store as dict (X, y):
# # =================================================================================================
def extract_embeddings(config, datas,
                       codebook_name,
                       recs_path=None):  # X: list of ndarrays of 1 x 1024. y: list of lists of strings of events names
    """ Output: dict for each dataset, each including of dictionaries: 
    X: list of 2D arrays for each recording
    y: list of string classes for each event.
    group: list rec ids
    """
    print("Extract embeddings for each vocalization for ecah recording for each dataset")
    # Create class:
    extract_emb_cls = ExtractEmbeddings([], config, codebook_name)
    extract_emb_cls.recs_path = recs_path
    embeddings_datas = {}
    for data_name, data in datas.items():
        extract_emb_cls.data_df = data
        ### Extract embeddings for each dataset: X: num of chunks of 0.975sX1024 for each voc (after trim/pad).
        X, y, _, group = extract_emb_cls.generate_embeddings()
        # Store in dict of the dataset:
        embeddings_datas[data_name] = {"X": X, "y": y, "group": group}

    return embeddings_datas


# Convert y to 2D dummy array(column of 0/1 for each class), and X to 2D array (dict to array): 
# # =================================================================================================
def convert_Xarr_yarr(config, datas, convert2dummy=True):
    print("For each dataset: Convert dict X to 2D array, and string labels y to dummy array")
    ### # Convert dict X to 2D array, and string labels y to dummy array:
    new_data = {}
    prepare_data_cls = PrepareData_Model(mapping=config["mapping_class"],
                                         convert2dummy=(config[
                                                            "act_out"] == "softmax"))  # True for sogtmax and flase for sigmoid
    for data_name, data in datas.items():
        new_data[data_name] = dict()
        new_data[data_name] = prepare_data_cls.execute(datas[data_name])

    return new_data


# Expand dimensions of X to 3D: keep y the same
# # =================================================================================================
def expand_X_dims(config, datas, models_expand=["conv1d_mdl"]):
    # expand dims of X for spesific models architectures:
    # Assuming X is a NumPy array
    if config["model_arc"] in models_expand:
        if datas["Train"]["X"].ndim != 3:
            print("Expanding dimensions of X for each dataset")
            new_data = {}
            for data_name, data in datas.items():
                new_data[data_name]["X"] = np.expand_dims(data_name["X"],
                                                          axis=-1)  # (n_smaples, 1, n_features, 1)
                new_data[data_name]["y"] = data_name["y"]
                new_data[data_name]["group"] = data_name["group"]

            return new_data
        else:
            return datas
    else:
        return datas


# Normalize each feature of the whole train data and then use the parameters to normalize test
# =================================================================================================
def norm_data_by_train(datas_embeddings, norm_type='zscore'):
    if (norm_type == 'none') or (norm_type is None):
        print('No embeddings normalization were applied')
        return datas_embeddings

    norm_embeddings_datas = datas_embeddings.copy()

    ### Define scaler:
    if norm_type == 'zscore':
        scaler = StandardScaler()
    elif norm_type == "min_max":
        scaler = MinMaxScaler()

    ### Normalize train data: If a variance is zero, can’t achieve unit variance, and the data is left as-is, giving a scaling factor of 1
    norm_embeddings_datas["Train"]["X"] = scaler.fit_transform(norm_embeddings_datas["Train"]["X"])
    ### Normalize test data using the parameters of train:
    norm_embeddings_datas["Test"]["X"] = scaler.transform(norm_embeddings_datas["Test"]["X"])

    return norm_embeddings_datas


# Normalize each embeddings vector to zero mean and unit variance:
# =================================================================================================
def norm_each_embedding(datas_embeddings, norm_type='zscore'):
    if norm_type == 'none':
        print('No embeddings normalization were applied')
        return datas_embeddings

    norm_embeddings_datas = {}
    for data_name, data in datas_embeddings.items():
        norm_embeddings_datas[data_name] = data.copy()
        if norm_type == 'zscore':
            norm_embeddings_datas[data_name]["X"] = stats.zscore(data["X"],
                                                                 axis=-1)  # NORMALIZE EACH EMBEDDING VECTOR TO BE ZERO MEAN AND UNIT VARIANCE
        elif norm_type == 'max':
            norm_embeddings_datas[data_name]["X"] = normalize(data["X"], norm='max',
                                                              axis=-1)  # norm each feature vector
        # Check:
        mu = np.mean(norm_embeddings_datas[data_name]["X"], axis=-1)
        sd = np.std(norm_embeddings_datas[data_name]["X"], axis=-1, ddof=1)
        print(f"Embeddings of {data_name} have Mean of [{min(mu):.3f}, {max(mu):.3f}]")
        print(f"Embeddings of {data_name} have SD of [{min(sd):.3f}, {max(sd):.3f}]")
    return norm_embeddings_datas


# =================================================================================================
def search_params(config, datas_dict, save_path, fold="fold0"):
    tuning_params = config["tuning_params"].copy()
    model_config = config["model_config"].copy()
    augment_config = config["augment_config"].copy()
    embeddings_config = config["feature_extraction"]

    if model_config["model_arc"] == "knn":
        if not tuning_params["k_neighbors"]:  # if empty
            # Set the numbe rof neighbors to tune:  
            tuning_params["k_neighbors"] = list(range(1, np.sqrt(datas_dict["Train"].shape[0]) + 1))  # sqrt(N_samples)
        else:  # define the given number as the maximum neighbors to check
            tuning_params["k_neighbors"] = list(range(1, tuning_params["k_neighbors"] + 1))

    save_path_fold = "{}/{}".format(save_path, fold)
    if not os.path.isdir(save_path_fold):
        os.makedirs(save_path_fold)

    search_results_cls = MyParamsSearch(config={**model_config,
                                                **embeddings_config,
                                                **augment_config},
                                        data_train=datas_dict["Train"],
                                        search_params=tuning_params,
                                        save_path=save_path_fold,
                                        n_folds=config["data_split_config"]["k_fold"] - 1)

    if ("read" in tuning_params["evaluate"]) & bool(config["model_config"].get("best_params",{})):  # read from yaml file:
        print('*Reading best model parameters from the yaml file*')
        search_results = pd.DataFrame()
        best_params = config["model_config"].get("best_params",{})
        # Create a new dictionary with the selected keys:
        best_params = {key: best_params[key] for key in list(tuning_params.keys()) if
                       key in best_params.keys()}
        print(f"The best_paramas read from the yaml file are: {best_params}")
        return search_results, best_params  # dataframe, dict

    elif "tune" in tuning_params["evaluate"] or \
            ("load" in tuning_params["evaluate"] and not os.path.isfile(
                save_path_fold / Path('Mean_CV_results.csv'))):  # if the Excel file doesnt exist
        print('*Tuning parameters*')
        search_results_cls.run_search_params()

    elif "load" in tuning_params["evaluate"]:  # load from the excel file:
        print('*Loading model parameters from the Hyper_tune folder*')
        search_results_cls.load_results()  # Load the parameters from the Excel file
        ### Check if the loaded combinations = n_combs:
        if len(search_results_cls.combinations) != search_results_cls.df_results.shape[0]:
            print('*Number of done combinations in excel file != target #combinations ->>> Tuning parameters*')
            # Continue tuning (and dont start from the beginning):
            search_results_cls.idx_comb = int(search_results_cls.df_results.shape[0]) + 1
            search_results_cls.run_search_params()

    # Get the table of results of all parameters combinations:
    search_results = search_results_cls.df_results
    # Get the best parameters (the first row in the table):
    best_params = search_results_cls.get_best_params()
    # Create a new dictionary with the selected keys:
    best_params = {key: best_params[key] for key in list(tuning_params.keys()) if
                   key in best_params.keys()}  # 07.11.23.

    return search_results, best_params  # dict 


# =============================================================================================
def save_model_fun(save_path, model):  # 08.03.22
    model_json = model.to_json()  # 09.02.2021
    with open(save_path / Path('Model.json'), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_path / Path('Model_weights.h5'))


# =============================================================================================
def check_load_trained_model(save_path):
    flag_train = True  # Flag that controls if to train the model or load trained

    # Check if the trained model exists:
    if os.path.isfile(rf"{save_path}/Model.json"):
        # Load the trained model:
        json_file = open(rf"{save_path}/Model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        print(f"{save_path}/Model.json was loaded")
        # Check if the weights of the model exist:
        if os.path.isfile(rf"{save_path}/Model_weights.h5"):
            # Load the model's weights:
            model = model_from_json(loaded_model_json)
            model.load_weights(rf"{save_path}/Model_weights.h5")
            model.summary()
            print(f"{save_path}/Model_weights.h5 was loaded")
            flag_train = False
            return model, flag_train
        else:
            return [], flag_train
    else:
        return [], flag_train


# Fine tune and evaluate the model
# # =================================================================================================
def compile_train_test_model(config, datas_dict, df_vocs_info, save_path, fold_full="fold0"):
    save_path_fold = rf"{save_path}/{fold_full}"
    flag_train = True  # Flag that controls if to train the model or load trained

    if config.get('use_loaded', False):
        model, flag_train = check_load_trained_model(save_path=save_path_fold)

    if flag_train:
        # Create the save_path_fold folder if does not exist already:
        if not os.path.isdir(save_path_fold):
            os.makedirs(save_path_fold)

        # Apply augmentation to train dataset:
        if config["if_augment"]:
            print('Apply SMOTE augmentation to train embeddings')
            # Apply SMOTE to the training data
            augmentor = AugmentEmbeddings(config, datas_dict["Train"])
            X_resampled, y_resampled = augmentor.X_resampled, augmentor.y_resampled
            datas_dict["Train"] = {'X': X_resampled, 'y': y_resampled}

        #  Train the classification model:
        mdl_cls = RunModel(config, datas_dict["Train"])
        history = mdl_cls.execute()
        model = mdl_cls.model
        if mdl_cls.config["model_arc"] not in mdl_cls.not_NN_classifiers:
            # Save model summary to txt file:
            with open(save_path_fold / Path('Model_summary.txt'), 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Save the model:
        if config["save_model"]:
            save_model_fun(save_path_fold, mdl_cls.model)

        # Plot the loss function:
        if history:
            fig_loss = plot_loss(history, title='')
            fig_loss.savefig(save_path_fold / Path("Loss_fig.png"), dpi=300, bbox_inches='tight')

    # Evaluate the trained model on test dataset:
    results, true_pred_df = evaluate_model(datas_dict, model, config, save_path_fold, data_name="Test")

    # Evaluate the trained model on train dataset:
    _, true_pred_df_train = evaluate_model(datas_dict, model, config, save_path_fold, data_name="Train")

    return results, true_pred_df  # dict, df


# -------------------------------------------------------------------------------------------------
def evaluate_model(data, model, config, save_path_fold, data_name="Test"):
    eval_cls = EvaluateModel(model, data[data_name],
                             class_names=list(set(config["mapping_class"].values())))
    eval_cls.model_extractor = config["embeddings_config"]["model_extractor"]
    eval_cls.run_type = config.get("run_type", "translearn")
    eval_cls.execute()
    K.clear_session()
    # Save the predicted and actual classes:
    if config.get("save_true_pred_df", True):
        eval_cls.true_pred_df.to_csv(save_path_fold / Path(f"True_pred_group_{data_name}.csv"), index=False)
    return eval_cls.results, eval_cls.true_pred_df


# Write to txt file the : train size, test size
# =================================================================================================
def save_data_size_results(datas_dict, save_path, filename="Data_size_results"):
    # Create the save_path_fold folder if doesnt exist already:
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    filename_save_path = f"{save_path}/{filename}.csv"
    # Create dict with all the columns:
    row = {"Train samples": datas_dict["Train"]["X"].shape[0],
           "Test samples": datas_dict["Test"]["X"].shape[0]}
    with open(filename_save_path, 'a', newline='') as csv_file:
        fieldnames = row.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # If the file is empty, write the header
        if csv_file.tell() == 0:
            writer.writeheader()
        # Write the data
        writer.writerow(row)


# Save dict of dicts to csv file: results of all folds
# =================================================================================================
def save_dict_of_dicts_to_csv(results_dict, main_save_path, filename="Classification_Results"):
    # Get today's date in format ddmmyy:
    today_date = date.today().strftime("%d%m%y")

    # Create full filename path and add today's date at the end:
    file_name_path = f"{main_save_path}/{filename}_{today_date}.csv"

    with open(file_name_path, 'w', newline='') as csvfile:  # Create csv file
        writer = csv.writer(csvfile)

        # Write the header row with keys of the inner dictionaries
        header = list(results_dict[next(iter(results_dict))].keys())
        writer.writerow([''] + header)

        # Write the data rows
        for outer_key, inner_dict in results_dict.items():
            row = [outer_key] + [inner_dict[key] for key in header]
            writer.writerow(row)
    print(f"Done saving all results to csv file in {file_name_path}")


# Plot loss function of a trained model
# =================================================================================================
def plot_loss(history, title=''):
    plt.figure(figsize=(8, 7))
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('#Epoch', fontsize=24, fontdict=FONT)
    plt.ylabel('Loss function', fontsize=24, fontdict=FONT)
    plt.title(title, fontsize=24, fontdict=FONT)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rc('font', **FONT)
    fig_loss = plt.gcf()
    plt.legend()
    return fig_loss


# =================================================================================================
def copy_script_folder(config, orig_path, dest_path,
                       extension='.py'):  # Copy the script to a folder in results
    if config.get('scripts_replace', True):
        destination_folder = f"{dest_path}/scripts"
    else:
        date_today = date.today().strftime("%d%m%y")
        destination_folder = f"{dest_path}/scripts_{date_today}"

    if not os.path.isdir(destination_folder):  # if doesnt exist
        os.mkdir(destination_folder)

    for filename in os.listdir(orig_path):
        if filename.endswith(extension):  # only files that ends with "extension"
            shutil.copy(f"{orig_path}/{filename}", f"{destination_folder}/{filename}")


# =============================================================================================
def copy_yaml_to_save_path(config, dest_path):
    # Copy the configuration file to the save path:
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    config_file = {"main_config": config}
    date_time_now = datetime.today().strftime('%d%m%Y_%H%M%S')
    with open(f"{dest_path}/config_{date_time_now}.yaml", 'w') as yaml_file:
        yaml.dump(config_file, yaml_file, default_flow_style=False)


# =================================================================================================
def get_codebook_name(config):
    if config["feats_type"].lower() == "pretrained":
        # Get the spectrograms configuration code name:
        CB_class = CreateCodebook(config=config["embeddings_config"],
                                  codebook_filename="PreTr_CodeBook")
        CB_class.execute()
        return CB_class.config_CB_name
    else:
        return ""


# =================================================================================================
def calc_classes_total_num_dur(vocs_df):  # Calculate number of vocs and their total duration per class
    num_dur_classes = {"Cry": {"Num": 0, "Dur": 0},
                       "Scream": {"Num": 0, "Dur": 0},
                       "Speech": {"Num": 0, "Dur": 0}}
    for rec_i, df in vocs_df.items():
        classes_rec = set(df["event"])
        for clas in classes_rec:
            df_cls = df[df["event"] == clas]
            num_dur_classes[clas]["Num"] += len(df_cls)
            num_dur_classes[clas]["Dur"] += sum(df_cls["length"]) / 60  # min
    return num_dur_classes


# =================================================================================================
def calc_num_dur_classes_per_rec(vocs_dict_df, save_path):  # 14.02.24.
    recs_num_classes = {}
    columns = []
    dur_per_class = {"Cry": [], "Scream": [], "Speech": []}
    for rec_name, rec_df in vocs_dict_df.items():
        rec_classes = rec_df["event"].unique()
        # print(f"************{rec_name}: ************")
        recs_num_classes[rec_name] = {"Cry": None, "Scream": None, "Speech": None,
                                      "Dur_Cry": None, "Dur_Scream": None, "Dur_Speech": None}
        for rec_clas in rec_classes:
            events_class_rec = rec_df[rec_df['event'] == rec_clas]
            num_events_cls_rec = len(events_class_rec)
            # print(f"Number of {rec_clas}: {num_events_cls_rec}")
            recs_num_classes[rec_name][rec_clas] = num_events_cls_rec
            recs_num_classes[rec_name][f"Dur_{rec_clas}"] = round(
                sum(events_class_rec['end'] - events_class_rec['start']), 2)
            dur_per_class[rec_clas].extend((events_class_rec['end'] - events_class_rec['start']).to_list())

        if not columns:
            columns = list(recs_num_classes[rec_name].keys())

    df = pd.DataFrame([(rec_name, *values.values()) for rec_name, values in recs_num_classes.items()],
                      columns=["rec_name", *columns])

    date_time_now = datetime.today().strftime('%d%m%Y_%H%M%S')
    df.to_excel(rf"{save_path}/Number_events_per_rec_per_class_{date_time_now}.xlsx", index=False)

    for cls_name, values in dur_per_class.items():
        average_duration = np.mean(values)
        std_duration = np.std(values, ddof=1)
        print(f"Average duration of {cls_name} = {average_duration:.3f}\u00B1{std_duration:.3f}")
    return df



# =================================================================================================
def get_rec_duration(audio_file_name):
    if not audio_file_name: # if None
        raise Exception("rec={audio_file_name} is None")
    # Open the audio file in read mode and get the number of frames and sample rate
    with sf.SoundFile(audio_file_name) as file:
        num_frames = len(file)
        sample_rate = file.samplerate
    # Calculate the duration of the audio file
    return num_frames / sample_rate


# calculate Concordance Correlation Coefficient
# -------------------------------------------------------------------------
def ccc_metric(y_true, y_pred):
    """
                Concordance Correlation Coefficient
    Pearson's r measures linearity, while CCC measures agreement.
    Imagine a scatterplot between the two measures:
        High agreement implies that the scatterplot points are close to the 45
        degrees line of perfect concordance which runs diagonally to the
        scatterplot, whereas a high Pearson's r implies that the scatterplot 
        points are close to any straight line.
    """
    N = len(y_true)  # K.shape(y_true)[0] # K.int_shape(y_pred)[-1]
    # print('y_pred shape = {}, N = {}'.format(y_pred.shape[0], N))
    # epsilon = np.finfo(float).eps
    # covariance between y_true and y_pred
    s_xy = np.dot((y_true - np.mean(y_true)), (y_pred - np.mean(y_pred))) / (N - 1.0)  # +epsion
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true, ddof=1)
    s_y_sq = np.var(y_pred, ddof=1)

    # condordance correlation coefficient
    # print('s_xy = {}, s_x_sq = {}, s_y_sq = {}, x_m = {}, y_m = {}'.format(
    #     s_xy, s_x_sq, s_y_sq, x_m, y_m))
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)
    if ccc > 1:
        print(f"CCC > 1: {ccc:.3f}")
    return ccc


# Calculate threshold where Precision=Recall
# =================================================================================================
def calcualte_best_th(labels, predictions, pos_label=0):
    print("Calcualte best threshold")
    precision, recall, ths = precision_recall_curve(labels,
                                                    predictions,  # probability estimates of the positive class
                                                    pos_label=pos_label)  # positive class
    # Calculate absolute difference between precision and recall
    diff = np.abs(precision - recall)
    # Find index of minimum difference
    min_diff_index = np.argmin(diff)
    # Retrieve the threshold corresponding to the minimum difference
    print(
        f"th={ths[min_diff_index]:.3f} -> Class {pos_label}: Recall={recall[min_diff_index]:.3f}, Precision={precision[min_diff_index]:.3f}")

    return 1 - ths[min_diff_index]


# run main
# =================================================================================================
# =================================================================================================
# =================================================================================================
if __name__ == "__main__":
    tic_toc.tic()
    print("Start main")
    ### Load configurations from yaml:
    options, main_config = extract_config()

    ### Define the GPU to work on:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(main_config.get('GPU_id'))

    if not main_config["feature_extraction"]["config_CB_name"]:
        ### Get embeddings configuration codebook name:
        codebook_name = get_codebook_name(main_config["feature_extraction"])
    else:
        codebook_name = main_config["feature_extraction"]["config_CB_name"]
    print(f"Codebook name: {codebook_name}")

    if not main_config["feature_extraction"]["embeddings_config"]["man_segs"]:
        ### Load vocal segments (or create if do not exist):
        vocal_segments_dict_df = extract_vocalizations(main_config)  # dict, of dataframe per recording
    else:
        ### Load manual segments:
        vocal_segments_dict_df = extract_manual_segments(main_config)
    print(
        f"Numer of vocs and their total duration of vocs before trim/pad: \n{calc_classes_total_num_dur(vocal_segments_dict_df)}")

    ### Convert dict to dataframe, where each row is an event:
    vocal_segments_df = vocal_segments_to_df(vocal_segments_dict_df)

    ### Split whole data to train-validation-test: dataframe for each dataset
    train_dict, test_dict = split_data_train_test_ktimes(main_config["data_split_config"],
                                                         vocal_segments_df)
    data_splits_folds = {"Train": train_dict, "Test": test_dict}

    ### Extract embeddings for the whole dataset:
    # group: list of integers of 1:num_recs
    emb_data_local = extract_embeddings(main_config["feature_extraction"],
                                        {"All": vocal_segments_df}, codebook_name)

    ### Create the main save path:
    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    date_today = date.today().strftime("%d%m%y")
    main_run_save_path = r"{}/models/{}/{}/{}/{}".format(
        get_database_path('results'), main_config["rec_list_yaml"].split(".")[0],
        main_config["model_config"]["model_arc"],
        main_config["feature_extraction"]["normalization"],  # Feature normalization create additional folder inside:
        codebook_name)
    # Add at the end additional information if needed:
    main_run_save_path = rf"{main_run_save_path}/{main_config['run_name']}" if main_config[
        "run_name"] else main_run_save_path
    # In case of using only part of the Speech data (10%):
    n_perc_keep_event = main_config["feature_extraction"]["train_prec_take"]
    main_run_save_path = rf"{main_run_save_path}/UnderSample{n_perc_keep_event}" \
        if (n_perc_keep_event < 100) else main_run_save_path
    # In case of augmentation create additional folder inside:
    main_run_save_path = rf"{main_run_save_path}/{main_config['augment_config']['augment_type']}" \
        if main_config["augment_config"]["if_augment"] else main_run_save_path

    # ---------------------------------------------------------------------------------------------   
    # ---------------------------------------------------------------------------------------------
    local_vocs_df = vocal_segments_dict_df.copy()

    # Create the main_run_save_path folder if doesnt exist already:
    if not os.path.isdir(main_run_save_path):
        os.makedirs(main_run_save_path)
        print("main_run_save_path was created: {}".format(main_run_save_path))
    else:
        print("main_run_save_path: {}".format(main_run_save_path))

    ### Copy the configuration file to the save path:
    copy_yaml_to_save_path(main_config, main_run_save_path)

    ### Copy the script used here to the save path:
    copy_script_folder(config=main_config, orig_path=mainpath,
                       dest_path=main_run_save_path, extension='.py')

    # Calculate and save number of vocs per class per recording:
    df_num_dur_vocs_per_rec = calc_num_dur_classes_per_rec(local_vocs_df,
                                                           main_run_save_path)

    #% For each fold: hyper-tune the parameters-> train a model with the best parameters-> evaluate on test dataset
    results_folds = {}
    grid_results = {}
    alg_man_rel_event_dur_r_p = {}
    p_values = {}
    
    plt.figure(figsize=(6, 5))
    
    ### For each fold:
    for i_fold, fold in enumerate(train_dict):

        embeddings_train_data, embeddings_test_data = {}, {}
        for key in emb_data_local["All"].keys():  # [X,y,group]
            # Take recordings names that in train of spesific fold:
            train_recs_names = train_dict[fold]["Recording"].unique().tolist()
            emb_data_local_train = {rec_name: emb_data_local["All"][key][rec_name]
                                    for rec_name in emb_data_local["All"][key]
                                    if rec_name in train_recs_names}

            # Take recordings names that in test of spesific fold:
            test_recs_names = test_dict[fold]["Recording"].unique().tolist()
            emb_data_local_test = {rec_name: emb_data_local["All"][key][rec_name]
                                   for rec_name in emb_data_local["All"][key]
                                   if rec_name in test_recs_names}

            embeddings_train_data[key] = emb_data_local_train.copy()
            embeddings_test_data[key] = emb_data_local_test.copy()

        emb_datas = {"Train": embeddings_train_data.copy(),
                     "Test": embeddings_test_data.copy()} # V

        """ Convert dict X to 2D array, and string labels y to dummy array:
        Output: dict for each dataset with X and y numpy arrays: 
        X: array of (num_vocs, num_features)
        y: array of (num_vocs, 3) or (num_vocs, )
        group: array of (num_vocs, ) """
        emb_datas_arr = convert_Xarr_yarr(main_config["model_config"], emb_datas)

        ### Expand dims of X for specific models architectures:
        emb_datas_arr = expand_X_dims(main_config["model_config"], emb_datas_arr) # V

        ### Normalize data:
        norm_emb = norm_data_by_train(datas_embeddings=emb_datas_arr,
                                      norm_type=main_config["feature_extraction"]["normalization"])

        ### Save Train and Test data sizes to csv file:
        save_data_size_results(datas_dict=norm_emb,
                               save_path="{}/{}".format(main_run_save_path, fold),
                               filename="Data_size_results")

        ### Change the tuning params to load ready parameters when need to use loaded model (no need to tune): 06.02.24.
        main_config["tuning_params"]["evaluate"] = ['load'] if main_config["model_config"].get('use_loaded', False) \
            else main_config["tuning_params"]["evaluate"]
        
        # ------------------------------------------------------------------------------------------
        ### DROP DUPLICATE ROWS IN TEST ONLY
        # Identify duplicate rows
        duplicate_mask = test_dict[fold].duplicated(keep='first')
        # Get the indexes of duplicate rows
        duplicate_indexes = test_dict[fold].index[duplicate_mask].tolist()
        not_duplicate_indexes = test_dict[fold].index[duplicate_mask==False].tolist()
        print(duplicate_indexes)
        new_test = []
        if duplicate_indexes:
            # Drop duplicate rows        
            test_dict[fold].drop(duplicate_indexes, inplace=True)
            unique_rows = np.unique(norm_emb["Test"]["X"], axis=0)
            print(f"#unique rows={unique_rows.shape[0]}, #orig rows = {norm_emb['Test']['X'].shape[0]}")
            new_test = {key: val[not_duplicate_indexes] for key,val in norm_emb["Test"].items()}
            norm_emb["Test"] = new_test.copy()
        # ------------------------------------------------------------------------------------------
        
        if len(norm_emb['Train']['y'].shape) <= 1:
            print(f"Number of samples in Train: {np.bincount(norm_emb['Train']['y'])}")
            print(f"Number of samples in Test: {np.bincount(norm_emb['Test']['y'])}")
        else:
            print(f"Number of samples in Train: [{np.sum(norm_emb['Train']['y'], axis=0)}]")
            print(f"Number of samples in Test: [{np.sum(norm_emb['Test']['y'], axis=0)}]")

        if train_dict[fold].shape[0] != norm_emb['Train']['y'].shape[0]:
            print(
                f"!!!!! train_dict[fold].shape[0] ({train_dict[fold].shape[0]}) != norm_emb['Train']['y'].shape[0] ({norm_emb['Train']['y'].shape[0]}) !!!!!")
            break
        
        
        ### Apply search for hyper-parameters:
        search_results_fold, best_params_fold = search_params(config=main_config,
                                                              datas_dict=norm_emb,
                                                              save_path=main_run_save_path,
                                                              fold=fold)
        # Keep only N random events in Train data:
        norm_emb_train_reduced, train_df_vocs = keep_N_random_events(
            norm_emb["Train"],
            n_p_keep=main_config["feature_extraction"]["train_prec_take"],
            class_label=main_config["model_config"]["mapping_class"]["Speech"],
            df_vocs=train_dict[fold])
        norm_emb_new = {"Train": norm_emb_train_reduced, "Test": norm_emb["Test"]}
    
        
        ### Train on the best parameters and evaluate the test dataset:
        results_folds[fold], df_true_pred = compile_train_test_model(config={**main_config["feature_extraction"],
                                                                             **main_config["model_config"],
                                                                             **main_config["augment_config"],
                                                                             **best_params_fold},
                                                                     datas_dict=norm_emb_new,
                                                                     df_vocs_info=train_df_vocs,
                                                                     save_path=main_run_save_path,
                                                                     fold_full=fold)  # results: a dict
        ### Merge two dictionaries (in order to save both the results and the chosen hyperparameters for each fold)
        results_folds[fold].update(best_params_fold)        
        
        
        if df_true_pred.shape[0] != test_dict[fold].shape[0]:
            print(
                f'df_true_pred shape: {df_true_pred.shape[0]} != test_dict[fold] shape {test_dict[fold].shape[0]}')
            continue

    ### Save classification of test results to csv file:
    save_dict_of_dicts_to_csv(results_dict=results_folds, main_save_path=main_run_save_path)
    print(f"****The whole process took {tic_toc.toc() / 60 / 60} hours.****")
