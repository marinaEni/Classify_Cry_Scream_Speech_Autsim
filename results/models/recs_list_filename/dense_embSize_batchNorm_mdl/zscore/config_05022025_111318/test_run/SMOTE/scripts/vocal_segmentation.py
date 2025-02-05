"""
13.01.2021
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy

from common import get_database_path, load_wav, calc_seg_energy, round_to_int, get_num_digits_after_point
import pandas as pd
from unite_labels import unite_labels


class VocalSegmentation:
    def __init__(self, rec_id, config, man_seg_path=''):
        self.rec_id = rec_id
        self.config = config
        self.fs = None
        self.signal_df = None  # dataframe: columns=["time", "signal"]
        self.frame_length = self.config.get('frame_length', 0.04)
        self.frame_step = self.config.get('frame_step', 0.01)
        self.man_seg_path = get_database_path('man_seg') if not man_seg_path else man_seg_path

        self.man_seg_united = None
        self.voc_seg = list()  # initialized as list of dictionaries.
        self.time_round_d = 2  # number of digits to remain in times of the segments
        self.voc_segs_columns = ['start', 'end', 'speaker', 'event']
        self.speakers = config.get('speakers', ['Child'])
        self.speakers2map = config["speakers2map"]
        self.recs_path = None # the path where the recordings placed. if none-> takes default from "common.py"

    # =============================================================================================
    def execute(self):
        print("*unite manually segments labels")
        self.man_seg_united = self.unite_man_labels()
        print("*map speakers labels")
        self.map_speakers()
        print("*load wav")
        self.signal_df, self.fs = load_wav(self.rec_id, rec_path=self.recs_path)
        print("*gen_vocal_segmentation")
        self.gen_vocal_segmentation()
        print("*save_voc_segs to csv")
        self.save_voc_segs_csv()
        print('*done creating vocal segments for a recording')

    # =============================================================================================
    def unite_man_labels(self):  # 06.09.22
        """Unite adjacent and continuous labels of the same speaker"""
        # Read the excel file:
        man_seg = pd.read_excel(Path(self.man_seg_path) / Path(self.rec_id + ".xlsx"),
                                header=None)
        # man_seg = pd.read_csv(Path(self.man_seg_path) / Path(self.rec_id + ".txt"),
        #                         header=None, sep="\t")
        
        # DROP DUPLICATE ROWS: 26.11.24.
        man_seg.drop_duplicates(inplace=True)
        
        # Determine the number of columns in the loaded DataFrame
        num_columns = man_seg.shape[1]
        # Check if the number of columns is less than the desired number
        if num_columns < len(self.voc_segs_columns):
            # Assign method for adding columns:
            for i in range(num_columns, len(self.voc_segs_columns)):
                man_seg = man_seg.assign(**{self.voc_segs_columns[i]: ''})

        man_seg.columns = self.voc_segs_columns
        # Unite adjacent labels of same speakers
        man_seg_united = unite_labels(man_seg)
        return man_seg_united

    # =============================================================================================    
    def map_speakers(self, df=pd.DataFrame()):  # 25.03.2023
        """ Map the spekaers to their new name """
        for key, value in self.speakers2map.items():
            if df.empty:
                self.man_seg_united.replace(key, value, inplace=True)
            else:
                df.replace(key, value, inplace=True)
        return df

    # =============================================================================================    
    def gen_vocal_segmentation(self):
        """
        Create vocal segments for each annotated segment for all the listed speakers.
        """
        # Remove DC from the whole signal: 10.09.22.
        self.signal_df['signal'] = self.signal_df['signal'] - np.mean(self.signal_df['signal'])
        idx_speaker = 0
        # For each labeled segment:
        for idx in self.man_seg_united.index:
            i_start = round_to_int(self.man_seg_united.loc[idx, 'start'] * self.fs, d=0)  # 12.09.22: added round_to_int
            i_end = round_to_int(self.man_seg_united.loc[idx, 'end'] * self.fs,
                                 d=0)  # 06.09.22: removed -1 because in python the last index not taken.
            speaker = self.man_seg_united.loc[idx, 'speaker']
            # If its not the wanted speaker then continue to the next labeled segment:
            if speaker not in self.speakers:
                continue
            segment = self.signal_df.loc[i_start:i_end - 1,
                      'signal']  # 10.09.22: removed -1 from i_start, because in dataframe.loc the last idx is included. Do not reset index because they use as time
            # if the segment is too short (shorter than a frame size) then continue to the next segment. 01.02.2023
            if segment.shape[0] < self.frame_length * self.fs:
                continue
            idx_speaker += 1
            # Calculate the energy thresholds:
            th1, th2 = self.calc_th1_th2(i_start=i_start, i_end=i_end)
            # Using the thresholds split the labeled segment to vocalizations:
            self.man_seg_to_voc_seg(segment=segment,
                                    start_time_seg=self.man_seg_united.loc[idx, 'start'],
                                    th1=th1, th2=th2, speaker=speaker,
                                    event=self.man_seg_united.loc[idx, 'event'])
            print('Event:{}, Speaker: {}'.format(idx_speaker, speaker))

        self.voc_seg = pd.DataFrame.from_dict(self.voc_seg)  # 'start', 'end', 'speaker'
        # print(self.voc_seg)
        print('Rec {} has {} vocalizations'.format(self.rec_id, len(self.voc_seg)))

    # =============================================================================================
    def calc_th1_th2(self, i_start, i_end, t_longer=20):
        """
        :param i_start: start time of the segment [samples]
        :param i_end: end time of the segment [samples]
        :param t_longer: time*2 extension of the longer segment [sec]
        :return: th1, th2: the thresholds values that define the start and end of the vocal segments
        """
        # tic_toc.tic()
        # take t_longer back :
        l_start = max(
            [0, i_start - (t_longer * self.fs)])  # 10.09.22 Replaced 1 with 0. 12.09.22: deleted int(np.floor(
        # take t_longer forward:
        l_end = min([i_end + (t_longer * self.fs) - 1, self.signal_df.shape[0]])  # 12.09.22. deleted int(np.ceil(
        seg = self.signal_df.loc[l_start: l_end - 1, 'signal'].reset_index(
            drop=True)  # 06.09.22: reset index. 10.09.22: added -1 because in dataframe.loc the last idx is included
        energy_seg = calc_seg_energy(signal=seg, fs=self.fs,
                                     frame_length=self.frame_length, frame_step=self.frame_step)
        # Calculate baseline of the longer segment:
        counts, bin_edges = np.histogram(energy_seg, bins=1000)  # ! the counts may be a little different from matlab
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        idx_center_max = np.argmax(counts)
        baseline = centers[idx_center_max]
        # Find start and ends of the vocal segments:
        th1 = 1.9 * baseline
        th2 = 1.1 * baseline
        # print("calc_th1_th2: {:.5f}".format(tic_toc.toc()))
        return th1, th2

    # =============================================================================================
    def man_seg_to_voc_seg(self, segment, start_time_seg, th1, th2, speaker, event):
        """
        Calculate vocal segments for one segment.
        :param segment: a signal segment, Series.
        :param th1: the threshold that defines the start of the vocal segment.
        :param th2: the threshold that defines the end of the vocal segment.
        :param speaker: label of the Speaker (Child, Therapist,...)
        :return: list of dictionaries with time stat, time end and speaker's label for each /
         detected vocal segment
        """
        # tic_toc.tic()
        t1 = int(self.config.get('t1', 50 * 1e-3) / self.frame_step)
        t2 = int(self.config.get('t2', 50 * 1e-3) / self.frame_step)
        # Calculate energy of the segment (not in dB):
        # print("Segment length {}, frame length  {}, frame step {}".format(segment.shape,
        #                                                                   self.frame_length,
        #                                                                   self.frame_step))
        energy_seg = calc_seg_energy(signal=segment, fs=self.fs, frame_length=self.frame_length,
                                     frame_step=self.frame_step)
        frames_start, frames_end = self.find_start_end_voc_seg(energy_seg=energy_seg,
                                                               th1=th1, th2=th2,
                                                               t1=t1, t2=t2,
                                                               state=0)  # no start, no end

        '''Because the algorithm firstly finds Start and afterwards the End, then if it doesnt find
        Start it wont search for the End. So, if it didn't find a Start then define it as the first
        sample and then try to find an End.'''
        # If not found Start than define the Start as the first frame and search for end:
        if not frames_start:
            frames_start, frames_end = self.find_start_end_voc_seg(
                energy_seg=energy_seg, th1=th1, th2=th2, t1=t1, t2=t2,
                state=2)  # 10.09.22: state was changed from 0 to 2
            frames_start = [0]
            # If not found End than define the End as the last frame:
            if not frames_end:
                frames_end = [len(energy_seg) - 1]  # 10.09.22: added -1
        # If it found Start but didn't find End for the last vocal segment:
        elif len(frames_start) - len(frames_end) == 1:
            frames_end.append(len(energy_seg) - 1)  # define the End as the last frame
        # If it didn't find End not only for the last vocal segment -> error in the function:
        else:
            assert len(frames_start) - len(frames_end) <= 1, \
                'start is larger than end by more than 1'
        # print("Start = {}, End = {}".format(frames_start, frames_end))
        # self.plot_voc_segs_eng_thresholds(energy_seg, th1, th2, frames_start, frames_end)

        # Convert samples to times:
        for idx in range(len(frames_start)):
            frame_start = frames_start[idx]
            frame_end = frames_end[idx]
            time_start_vs = frame_start * self.frame_step  # in sec
            time_end_vs = frame_end * self.frame_step  # in sec
            if frame_end == len(energy_seg) - 1:  # the finish is at the end of the segment
                time_end_vs = len(segment) / self.fs  # in sec

            # start_time_seg = (segment.index[0]/self.fs)
            self.voc_seg.append(
                {'start': round_to_int(start_time_seg + time_start_vs, d=self.time_round_d),
                 'end': round_to_int(start_time_seg + time_end_vs, d=self.time_round_d),
                 'speaker': speaker,
                 'event': event})
            # print('Vocal segment start time = {}, end time = {}'.format(
            #     round_to_int(start_time_seg + time_start_vs, d=self.time_round_d),
            #     round_to_int(start_time_seg + time_end_vs, d=self.time_round_d)))
        # print("man_seg_to_voc_seg: {:.5f}".format(tic_toc.toc()))

    # =============================================================================================
    @staticmethod
    def get_dir_path(config):
        return get_database_path('vocal_seg') / Path('vocal_segments') / \
               Path('frame_length_{}__frame_step_{}__t1_{}__t2_{}'.format(
                   config.get('frame_length', 0.05),
                   config.get('frame_step', 0.01),
                   config.get('t1', 0.05),
                   config.get('t2', 0.05)))

    # =============================================================================================
    def get_file_path(self) -> str:
        return self.get_dir_path(self.config) / Path('{}_vs.txt'.format(self.rec_id))

    # =============================================================================================
    def save_voc_segs_csv(self):
        """
        The function saves the vocal segments dataframe to txt file.
        """
        # tic_toc.tic()
        dir_path = self.get_dir_path(self.config)
        if not os.path.isdir(dir_path):  # TODO check why writed
            os.makedirs(dir_path)  # 27.04.22. + r'\vocal_segments')

        self.voc_seg.to_csv(self.get_file_path(), sep='\t', float_format='%.2f',
                            header=False, index=False)
        # print("save_voc_segs: {:.5f}".format(tic_toc.toc()))

    # =============================================================================================
    def exists(self):
        return os.path.isfile(self.get_file_path()) and (not self.config.get('generate'))

    # =============================================================================================
    @staticmethod
    def plot_voc_segs_eng_thresholds(energy_seg, th1, th2, frames_start, frames_end):
        """
        The function plots detected vocal segments for one segment.
        :param energy_seg: array of energy values of the segment's frames.
        :param th1: the threshold that defines the start of the vocal segment.
        :param th2: the threshold that defines the end of the vocal segment.
        :param frames_start: list of frames indexes that define start of vocal segments
        :param frames_end: list of frames indexes that define end of vocal segments
        """
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(10 * np.log10(energy_seg))
        ax.hlines(10 * np.log10(th1), xmin=0, xmax=len(energy_seg), colors='b', label='th1')
        ax.hlines(10 * np.log10(th2), xmin=0, xmax=len(energy_seg), colors='k', label='th2')
        ax.vlines(frames_start,
                  ymin=min(10 * np.log10(energy_seg)), ymax=max(10 * np.log10(energy_seg)),
                  colors='g', label='Start')
        ax.vlines(frames_end,
                  ymin=min(10 * np.log10(energy_seg)), ymax=max(10 * np.log10(energy_seg)),
                  colors='r', label='End')
        ax.set_ylabel('Energy [dB]')
        ax.set_xlabel('Frame #')
        ax.legend(ncol=2)
        plt.show()

    # =============================================================================================
    @staticmethod
    def find_start_end_voc_seg(energy_seg, th1, th2, t1=5, t2=5, state=0):
        """
        Parameters
        ----------
        energy_seg : matrix: [number_of_frames x frame_length]
            energy vector for each frame of.
        th1 : float
            Value where above it will define the Start of the vocal segment.
        th2 : float
            Value where below it will define the End of the vocal segment.
        t1 : int
            The number of frames to be above th1 to define the Start of the vocal segment.
            The default is 5.
        t2 : int
            The number of frames to be below th2 to define the End of the vocal segment.
            The default is 5.
        state : int
            The state from which to start the algorithm.
            state = 0: search for first time above th1
            state = 1: keep checking above th1 for t1 frames
            state = 2: found Start -> search for first time below th2
            state = 3: keep checking below th2 for t1 frames
            The default is 0.

        Returns
        -------
        frames_start : int
            array of frames indexes of vocal segments start points.
        frames_end : int
            array of frames indexes of vocal segments end points.
        """
        # tic_toc.tic()
        count = 0
        frames_start = []
        frames_end = []
        for idx, eng_frame in enumerate(energy_seg):
            if state == 0:  # Search Start
                count = 0  # initialize the time counter
                if eng_frame >= th1:
                    # found first time above th1, move state 1 to see if it longs t1 frames:
                    state = 1
                    count = 1
                    frames_start.append(idx)  # save the Start point
            elif state == 1:  # keep checking above th1 for t1 frames
                if eng_frame >= th1:  # keep checking above th1
                    count += 1
                    if count == t1:  # check if the required period had pass
                        state = 2  # move for state that searches end
                else:  # Not above th1 -> delete the saved Start and start searching for Start again
                    state = 0
                    frames_start = frames_start[:-1]  # remove the last start
            elif state == 2:  # Search End:
                count = 0  # initialize the time counter below th2
                if eng_frame <= th2:
                    state = 3  # found first time above th1, move to next state
                    count = 1
            elif state == 3:  # keep checking below th2 for t2 frames
                if eng_frame <= th2:  # keep checking above th2
                    count += 1
                    if count == t2:  # check if the required period had pass
                        state = 0  # found End, start searching for new Start / new vocal segment
                        frames_end.append(idx)  # save the final point below th2
                else:  # Not above th2-> find new End
                    state = 2
        # print("find_start_end_voc_seg: {:.10f}".format(tic_toc.toc()))
        return frames_start, frames_end

    # =============================================================================================
    def load_vs(self, limit_start=None, limit_end=None):

        if limit_start == None:
            limit_start = self.config['limit_start']
        if limit_end == None:
            limit_end = self.config['limit_end']
        if limit_end == 'inf':
            limit_end = np.inf
        # Calculate the number of digits after the decimal point:
        d = get_num_digits_after_point(limit_start) + 1
        # Load vocal segments:
        # vs_file_path = r'{}\vocal_segments\{}_vs.txt'.format(
        # self.get_dir_path(self.config), self.rec_id)
        voc_segs = pd.read_csv(self.get_file_path(), sep='\t', header=None, index_col=None)
        voc_segs.columns = self.voc_segs_columns
        # map speakers labels
        voc_segs = self.map_speakers(df=voc_segs)  # 25.03.2023
        # Take only the segments that in the range:
        voc_segs['length'] = pd.Series([round_to_int(voc_seg['end'] - voc_seg['start'],
                                                     d=d) for _, voc_seg in voc_segs.iterrows()])
        # voc_segs['length'] = voc_seg['end']-voc_seg['start']  
        length_filter = np.logical_and(
            voc_segs['length'] >= limit_start,  # 13.09.22: deleted float
            voc_segs['length'] < limit_end)  # 13.09.22 deleted float
        voc_segs = voc_segs.loc[length_filter, :]
        voc_segs.reset_index(drop=True, inplace=True)

        # Take only the vocalizations of certain speakers: 23.01.2023
        voc_segs = voc_segs[voc_segs['speaker'].isin(self.speakers)]
        # 09.05.23: Take only the vocalizations of certain event type:
        if not (self.config["events_load"] == "all"):  # 06.10.2023.
            voc_segs = voc_segs[voc_segs["event"].isin(self.config["events_load"])]
        return voc_segs

    # =============================================================================================
    def save_vocs_event(self, voc_segs):
        ''' save each vocal segment as wav file as "<rec_id>_<event_name>_<#seg> '''
        print("*load wav")
        self.signal_df, self.fs = load_wav(self.rec_id)
        event_count = {event: 0 for event in self.config["events_save"]["list_events"]}
        for idx in voc_segs.index:
            voc_seg = voc_segs.loc[idx]
            # if its the wanted event type:
            if voc_seg["event"] in self.config["events_save"]["list_events"]:
                # create the save_path if doesnt exist:
                save_path = get_database_path("data_list") / \
                            self.config["events_save"]["save_path"] / \
                            voc_seg["event"]

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    # If its not the wanted speaker then continue to the next segment:
                if voc_seg['speaker'] not in self.speakers:
                    continue
                # create the signal:
                i_start = round_to_int(voc_seg['start'] * self.fs, d=0)  # 12.09.22: added round_to_int
                i_end = round_to_int(voc_seg['end'] * self.fs,
                                     d=0)  # 06.09.22: removed -1 because in python the last index not taken.
                voc_signal = self.signal_df.loc[i_start:i_end - 1, 'signal']

                event_count[voc_seg["event"]] += 1

                # save to wav file:
                file_savename = "{}\{}_{}_{}.wav".format(save_path,
                                                         self.rec_id,
                                                         voc_seg["event"],
                                                         event_count[voc_seg["event"]])
                # check if the file already exists:
                if not os.path.isfile(file_savename):
                    print("Save {} {} event as wav file".format(voc_seg["event"],
                                                                event_count[voc_seg["event"]]))
                    scipy.io.wavfile.write(file_savename, self.fs, voc_signal)
