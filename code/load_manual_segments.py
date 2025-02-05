# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:19:28 2021

@author: marinamu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from common import load_yaml, get_database_path, get_num_digits_after_point, round_to_int
from vocal_segmentation import VocalSegmentation

plt.rcParams.update({'font.family': 'Cambria'})
plt.rcParams['figure.dpi'] = 300  # Change the default dpi settings in matplotlib
plt.rcParams['figure.constrained_layout.use'] = True


class LoanManualSegments:

    def __init__(self, config, recs_names_list, man_seg_path='', should_print=True):
        self.config = config
        self.man_seg_path = man_seg_path

        if recs_names_list:  # if not empty
            rec_list_path = get_database_path('data_list') / recs_names_list
            self.recs_names_list = load_yaml(file_pointer=rec_list_path)
        else:
            self.recs_names_list = recs_names_list
        self.manual_segments = dict()  # dataframe for each recording
        self.should_print = should_print
        self.recs_path = None  # the path where the recordings placed. if none-> takes default from "common.py"
        self.speakers = config.get('speakers', ['Child'])
        self.speakers2map = {"ChildEcholalila": 'Child',
                             "Therapist2": "Therapist"}

    # =============================================================================================
    def print_if_needed(self, message):
        if self.should_print:
            print(message)

    # =============================================================================================
    def segments_lens(self):
        # max_length = 0
        self.lens = []
        for rec_name in self.recs_names_list:
            lens_rec = self.manual_segments[rec_name].end - self.manual_segments[rec_name].start
            self.lens.append(lens_rec.to_list())
    
    # =============================================================================================
    def get_man_segs_rec(self, rec_id):
        manual_segments = None
        
        vs_obj = VocalSegmentation(rec_id, self.config, self.man_seg_path)
        vs_obj.recs_path = self.recs_path  # 31.03.24.

        vs_obj.man_seg_united = vs_obj.unite_man_labels()
        # Map speakers names: ChildEcholalia->Child, Therapist2->Therapist
        vs_obj.map_speakers()

        manual_segments = vs_obj.man_seg_united.copy()
        # Take segments that only with certain length and of certain speakers
        manual_segments = self.filter_segments(man_segs=manual_segments)

        # Replace 'Echolalia' with 'Speech':
        if self.config["replace_echolalia2speech"]:
            manual_segments.replace({'Echolalia': 'Speech'}, inplace=True)
        return manual_segments
    
    # =============================================================================================
    def execute_config(self):
        
        # Load manual segments for each recording
        for idx, rec_name in enumerate(self.recs_names_list):
            self.print_if_needed("Loading manual segments for {}: {}/{}".format(
                rec_name, idx + 1, len(self.recs_names_list)))
            self.manual_segments[rec_name] = self.get_man_segs_rec(rec_name)

        if self.config["plot_statistics"]:
            # Plot vocal segments length of all recs in the list:
            self.plot_dist_lens()
            # Plot distribution of number of events:
            self.print_event_count()

    # =============================================================================================
    def filter_segments(self, man_segs, limit_start=None, limit_end=None):
        if limit_start is None:
            limit_start = self.config['limit_start']
        if limit_end is None:
            limit_end = self.config['limit_end']
        if limit_end == 'inf':
            limit_end = np.inf
        # Calculate the number of digits after the decimal point:
        d = get_num_digits_after_point(limit_start) + 1

        # Take only the segments of certain speakers: 23.01.2023
        man_segs = man_segs[man_segs['speaker'].isin(self.speakers)].reset_index(drop=True) # 02.09.24: added reset_index (influenced the length code row)

        # Take only the segments that in the range:
        man_segs.loc[:, 'length'] = pd.Series([round_to_int(voc_seg['end'] - voc_seg['start'],
                                                            d=d) for _, voc_seg in man_segs.iterrows()])
        # voc_segs['length'] = voc_seg['end']-voc_seg['start']
        length_filter = np.logical_and(
            man_segs['length'] >= limit_start,  # 13.09.22: deleted float
            man_segs['length'] < limit_end)  # 13.09.22 deleted float
        man_segs = man_segs.loc[length_filter, :]
        man_segs.reset_index(drop=True, inplace=True)

        # 09.05.23: Take only the segments of certain event type:
        if not (self.config["events_load"] == "all"):  # 06.10.2023.
            man_segs = man_segs[man_segs["event"].isin(self.config["events_load"])]
        return man_segs

    # =============================================================================================
    def plot_dist_lens(self):
        self.segments_lens()
        l = [item for sublist in self.lens for item in sublist]
        bins = np.linspace(0, np.int(np.ceil(np.max(l))), 100)
        mean = np.mean(l)
        p_95 = np.percentile(l, 95)  # return 95th percentile, e.g median.
        p_66 = np.percentile(l, 66)  # return 66th percentile, e.g median.
        characteristics = '\n#Rec = {}, Min = {:.3f}s, Mean = {:.3f}s,\n66% = {:.3f}s, 95% = {:.3f}s, Max = {:.3f}s\n'.format(
            len(self.lens), np.min(l), mean, p_66, p_95, np.max(l))
        self.print_if_needed(characteristics)

        plt.figure()
        plt.hist(l, bins, ec='black')
        yl, yh = plt.gca().get_ylim()
        plt.vlines(mean, ymin=yl, ymax=yh, label='Mean', colors='r', linestyles={'solid'})
        plt.vlines(p_66, ymin=yl, ymax=yh, label='66%', colors='g', linestyles={'dashed'})
        plt.vlines(p_95, ymin=yl, ymax=yh, label='95%', colors='k', linestyles={'dashdot'})
        plt.xlabel('Duration [sec]')
        plt.ylabel('Counts')
        plt.title(characteristics)
        plt.legend()
        plt.grid(True)
        plt.show()

    # =============================================================================================
    def print_event_count(self):
        concatenated_df = pd.concat(self.manual_segments.values(), ignore_index=True)
        # Add recording's name into a new column:
        concatenated_df['Recording'] = [key for key in self.manual_segments for i in
                                        range(len(self.manual_segments[key]))]
        # Group the event counts by recording and then compute the sum of the counts for each event type within each recording. This will give us a new DataFrame that has one row for each recording
        # and one column for each event type, with the count of each event type in that recording:
        event_counts = concatenated_df.groupby(['Recording', 'event']).size().reset_index(name='count')
        event_counts_by_recording = event_counts.groupby(['Recording', 'event'])[
            'count'].sum().reset_index()  # unstack(fill_value=0)
        sns.set_style("whitegrid")
        boxplot = event_counts_by_recording.boxplot(by='event', column=['count'], fontsize=12)
        return boxplot
