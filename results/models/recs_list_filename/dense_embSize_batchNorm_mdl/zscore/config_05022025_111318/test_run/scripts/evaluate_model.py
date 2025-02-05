# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:44:53 2023

@author: marinamu
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, cohen_kappa_score, roc_auc_score, balanced_accuracy_score
import pandas as pd
import numpy as np
import random

random.seed(10)


class EvaluateModel:
    def __init__(self, model, data_test, class_names=[0, 1, 2]):
        self.model = model
        self.data_test = data_test
        self.group = data_test.get("group", None)
        self.results = dict()
        self.true_pred_df = pd.DataFrame()
        self.class_names = class_names

        self.torch_models = ["whisper", "wav2vec"]
        self.model_extractor = "default"
        self.run_type = "translearn"
        self.th = 0.5  # threshold in binary classification

    # ---------------------------------------------------------------------------------------------                
    def execute(self):
        # Predict using trained model:
        predictions = self.model.predict(self.data_test["X"])
        from tensorflow.keras import backend as K
        # Clear the session to free up resources
        K.clear_session()

        # Create dataframe of the results: each row is event
        self.true_pred_df = self.create_pred_true_df(y_pred_prob=predictions,
                                                     group=self.group)
        print(f"shape of y_test: {self.data_test['y'].shape}")
        
        self.evaluate(y_true=self.data_test["y"], y_pred=predictions)
    
    # ---------------------------------------------------------------------------------------------                
    def evaluate(self, y_true, y_pred):
        if len(np.unique(self.class_names)) == 3:
            self.results = self.get_performance(y_true, y_pred)
        elif len(np.unique(self.class_names)) == 2:
            self.results = self.get_performance_2clas(y_true, y_pred)

    # =============================================================================================
    def create_pred_true_df(self, y_pred_prob, group):
        # number of probabilitity arrays:
        n_probs = y_pred_prob.shape[1]

        # Creating a DataFrame from the NumPy array
        df = pd.DataFrame(y_pred_prob, columns=[f'y_pred_{i}' for i in range(n_probs)])
        # Adding the y_true array as an additional column
        df['y_true'] = self.prob_to_class(self.data_test["y"])
        df['y_pred'] = self.prob_to_class(y_pred_prob)
        df['group'] = group
        return df

    # =============================================================================================
    def prob_to_class(self, output):
        # Return class labels (in case of probabilities find the argmax):
        if (len(output.shape) == 1) or (
                (len(output.shape) == 2) and (output.shape[1] == 1)):  # 2 class problem-> probability
            return self.convert_to_class(output, self.th)
        else:  # multi class
            return np.argmax(output, axis=-1)

    # =============================================================================================
    def get_performance(self, y_true, y_pred_prob):  # Assuming y_pred and y_test are (n_samples, )
        # Get the predicted classes using argamx:
        y_pred = self.prob_to_class(y_pred_prob)
        y_true = self.prob_to_class(y_true)

        # Caluclate confusion matrix:
        conf_mat = confusion_matrix(y_true, y_pred, labels=np.unique(y_true)).transpose().astype(
            dtype=np.int64)  # columns = true, rows = predicted. np.int64 important for MCC
        # Convert to DataFrame with rows names and columns names as class_names
        df_cm = pd.DataFrame(np.mat(conf_mat), index=self.class_names, columns=self.class_names)
        # Calc metrics:
        accuracy = accuracy_score(y_true, y_pred)*100  
        precision = precision_score(y_true, y_pred, average=None, zero_division=np.nan)*100  # ppv
        recall = recall_score(y_true, y_pred, average=None, zero_division=np.nan)*100  # sensitivity
        f1 = f1_score(y_true, y_pred, average=None, zero_division=np.nan)*100

        c_kappa = cohen_kappa_score(y_true, y_pred)
        auc = 100 * roc_auc_score(y_true, y_pred_prob, multi_class="ovr", average=None)  # area under curve. ovr=one vs others
        uar = 100 * balanced_accuracy_score(y_true, y_pred)  # area under curve. ovr=one vs others
        mcc = matthews_corrcoef(y_true, y_pred)

        # Print the evaluation metrics
        print("\nAUC:", auc)
        print(f"Unweightd Average Recall: {uar:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Cohen kappa Score: {c_kappa:.3f}")
        print("Matthews correlation coefficient:\n", mcc)
        print("Confusion Matrix:\n", df_cm)

        results = {'accuracy': accuracy,
                   'cohen_kappa': c_kappa,
                   'conf_mat': df_cm,
                   'uar': uar,
                   'mcc': mcc}
        # Add metrics per class:
        for idx, class_name in enumerate(self.class_names):
            results.update({f'f1 {class_name}': f1[idx]})
            results.update({f'recall {class_name}': recall[idx]})
            results.update({f'precision {class_name}': precision[idx]})
            if len(np.unique(y_true)) > 2:
                results.update({f'auc {class_name}': auc[idx]})
        if len(np.unique(y_true)) == 2:
            results.update({'auc': auc})

        return results

    # =============================================================================================
    @staticmethod
    def convert_to_class(y_pred_prob, th):
        # Create an array to store the class labels
        y_pred_class = np.zeros_like(y_pred_prob, dtype=int)

        # Assign class labels based on the threshold
        y_pred_class[y_pred_prob >= th] = 1

        return y_pred_class

    # =============================================================================================
    def get_performance_2clas(self, y_true, y_pred_prob):  # Assuming y_pred and y_test are (n_samples, )

        class_names = self.class_names[:2]  # the first two classes
        y_pred = self.convert_to_class(y_pred_prob, self.th)

        # Caluclate confusion matrix:
        conf_mat = confusion_matrix(y_true, np.squeeze(y_pred), labels=class_names).transpose().astype(
            dtype=np.int64)  # columns = true, rows = predicted
        # Convert to DataFrame with rows names and columns names as class_names
        df_cm = pd.DataFrame(np.mat(conf_mat), index=class_names, columns=class_names)

        # Perform predictions using the trained model
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)*100  
        precision = precision_score(y_true, y_pred, average=None, zero_division=np.nan)*100  # ppv
        recall = recall_score(y_true, y_pred, average=None, zero_division=np.nan)*100  # sensitivity
        f1 = f1_score(y_true, y_pred, average=None, zero_division=np.nan)*100

        
        c_kappa = cohen_kappa_score(y_true, y_pred)
        if len(set(y_true))>1:
            auc = 100 * roc_auc_score(y_true, y_pred_prob)  # area under curve.
        else:
            auc = None
        uar = 100 * balanced_accuracy_score(y_true, y_pred)     
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Print the evaluation metrics
        print("AUC:", auc)
        print(f"Unweightd Average Recall: {uar:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Cohen kappa Score: {c_kappa:.3f}")
        print("Matthews correlation coefficient:\n", mcc)
        print("Confusion Matrix:\n", df_cm)

        results = {'accuracy': accuracy,
                   'cohen_kappa': c_kappa,
                   'conf_mat': df_cm,
                   'auc': auc,
                   'uar': uar,
                   'mcc': mcc}

        for idx, class_name in enumerate(class_names):
            results.update({f'f1 {class_name}': f1[idx]})
            results.update({f'recall {class_name}': recall[idx]})
            results.update({f'precision {class_name}': precision[idx]})

        return results
