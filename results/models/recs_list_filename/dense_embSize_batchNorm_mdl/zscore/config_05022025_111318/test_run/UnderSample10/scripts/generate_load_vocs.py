# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:19:28 2021

@author: marinamu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from common import load_yaml, get_database_path
from vocal_segmentation import VocalSegmentation

plt.rcParams.update({'font.family': 'Cambria'})
plt.rcParams['figure.dpi'] = 300  # Change the default dpi settings in matplotlib
plt.rcParams['figure.constrained_layout.use'] = True

class GenerateLoadVocs:

    def __init__(self, config, recs_names_list, man_seg_path='', should_print=True):
        self.config = config
        self.man_seg_path = man_seg_path
        
        if recs_names_list:  # if not empty
            rec_list_path = get_database_path('data_list') / recs_names_list
            self.recs_names_list = load_yaml(file_pointer=rec_list_path)
        else:
            self.recs_names_list = recs_names_list
        self.vocal_segments = dict()
        self.should_print = should_print
        self.recs_path = None # the path where the recordings placed. if none-> takes default from "common.py"
    
    # =============================================================================================
    def print_if_needed(self, message):
        if self.should_print:
            print(message)

    # =============================================================================================
    def vs_lens(self):
        # max_length = 0
        self.lens = []
        for rec_name in self.recs_names_list:
            lens_rec = self.vocal_segments[rec_name].end - self.vocal_segments[rec_name].start
            self.lens.append(lens_rec.to_list())

    # =============================================================================================
    def execute_config(self):
        vs_obj = dict()
        missing_rec_vs_list = list()

        # ************************* Create missing lists **********************
        # Step: Check if exist:
        for rec_name in self.recs_names_list:
            vs_obj[rec_name] = VocalSegmentation(rec_name, self.config, self.man_seg_path)
            vs_obj[rec_name].recs_path = self.recs_path  # 31.03.24.
            
            # check if the file with the desired parameters already exist:                
            if not vs_obj[rec_name].exists():  # 28.11.21
                missing_rec_vs_list.append(rec_name)

        # ************************* Generate vocal segments for missing recs *************************
        # Step: create vocal segments for recordings missing vocal segments:
        for idx, rec_name in enumerate(missing_rec_vs_list):
            # vs_obj[rec_name] = VocalSegmentation(rec_name, self.vocal_segment_config)
            self.print_if_needed("Generating vocal segments for {}: {}/{}".format(rec_name, idx + 1,
                                                                                  len(missing_rec_vs_list)))
            vs_obj[rec_name].execute()

        # ************************* Load vocal segemnts for all recs *************************
        # Step: load vocal segments for all recordings:
        for idx, rec_name in enumerate(self.recs_names_list):
            self.print_if_needed('Loading vocal segments of {}: {}/{}'.format(rec_name, idx + 1,
                                                                              len(self.recs_names_list)))
            self.vocal_segments[rec_name] = vs_obj[rec_name].load_vs()  # dataframe
            # Replace 'Echolalia' with 'Speech':
            if self.config["replace_echolalia2speech"]:
                self.vocal_segments[rec_name].replace({'Echolalia': 'Speech'}, inplace=True)
                
            if self.config.get("events_save", {}).get("save_TF", False):
                vs_obj[rec_name].save_vocs_event(self.vocal_segments[rec_name])

        if self.config["plot_statistics"]:
            # Plot vocal segments length of all recs in the list:
            self.plot_dist_lens()
            # Plot distribution of number of events:
            self.print_event_count()

    # =============================================================================================
    def plot_dist_lens(self):
        self.vs_lens()
        l = [item for sublist in self.lens for item in sublist]
        bins = np.linspace(0, int(np.ceil(np.max(l))), 100)
        mean = np.mean(l)
        p_95 = np.percentile(l, 95)  # return 95th percentile, e.g median.
        p_66 = np.percentile(l, 66)  # return 66th percentile, e.g median.
        characteristics = '\n#Recs = {}, Min = {:.3f}s, Mean = {:.3f}s,\n66% = {:.3f}s, 95% = {:.3f}s, Max = {:.3f}s\n'.format(
            len(self.lens), np.min(l), mean, p_66, p_95, np.max(l))
        self.print_if_needed(characteristics)

        plt.figure()
        plt.hist(l, bins, ec='black', density=True)
        yl, yh = plt.gca().get_ylim()
        plt.vlines(mean, ymin=yl, ymax=yh, label='Mean', colors='r', linestyles={'solid'})
        plt.vlines(p_66, ymin=yl, ymax=yh, label='66%', colors='g', linestyles={'dashed'})
        plt.vlines(p_95, ymin=yl, ymax=yh, label='95%', colors='k', linestyles={'dashdot'})
        plt.xlabel('Duration [sec]')
        plt.ylabel('Probability')
        plt.title(characteristics)
        plt.legend()
        plt.grid(True)
        plt.show()

    # =============================================================================================
    def print_event_count(self):
        concatenated_df = pd.concat(self.vocal_segments.values(), ignore_index=True)
        # Add recording's name into a new column:
        concatenated_df['Recording'] = [key for key in self.vocal_segments for i in
                                        range(len(self.vocal_segments[key]))]
        # Group the event counts by recording and then compute the sum of the counts for each event type within each recording. This will give us a new DataFrame that has one row for each recording
        # and one column for each event type, with the count of each event type in that recording:
        event_counts = concatenated_df.groupby(['Recording', 'event']).size().reset_index(name='count')
        event_counts_by_recording = event_counts.groupby(['Recording', 'event'])[
            'count'].sum().reset_index()  # unstack(fill_value=0)
        sns.set_style("whitegrid")
        boxplot = event_counts_by_recording.boxplot(by='event', column=['count'], fontsize=12)
        return boxplot
