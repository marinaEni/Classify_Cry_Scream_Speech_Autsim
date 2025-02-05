import copy
import random
import yaml
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import soundfile as sf
from scipy.stats import norm

CODE_PATH = Path(__file__).parent.absolute()
os.chdir(CODE_PATH)
PROJECT_PATH = os.path.abspath(os.path.join(CODE_PATH, os.pardir))

# load_yaml
# =============================================================================
def load_yaml(file_pointer):
    if not file_pointer:
        raise ValueError("The config file path is not set or is None.")
    with open(file_pointer, 'r') as file:
        print("Load yaml file:")
        return yaml.safe_load(file) # dict


# get_database_path
# =============================================================================
def get_database_path(name):
    if name not in ['rec', 'man_seg','data_list', 'results', 'prefix', 'codebooks', 'embeddings']:
        assert 0, "Can't find path to {}".format(name)

    path_dict = dict()

    path_dict['prefix'] = r''
    path_dict['rec'] = r'data/recordings'
    path_dict['data_list'] = r'data/recs_lists'
    path_dict['man_seg'] = r'data/manual_annotations'
    path_dict['codebooks'] = 'codebooks'
    path_dict['results'] = 'results'
    path_dict['embeddings'] = r'results/embeddings'

    return PROJECT_PATH / Path(path_dict[name])

# =============================================================================
def load_wav(rec_id, rec_path=None, target_fs=16000, channel=0, return_df=True):
    """
    Load audio signal of a specific recording
    """
    ### 06.12.23
    if not rec_path:
        rec_path = get_database_path('rec')
    else:
        rec_path = PROJECT_PATH / rec_path

    file_path = rec_path / Path('{}.wav'.format(rec_id))
    fs, wav_data = wavfile.read(file_path, 'rb')
    if len(wav_data.shape) > 1:
        wav_data = wav_data[:, channel]
    if fs != target_fs:
        print(f"The sampling rate of {file_path} !=16kHz, but {fs}")
    
    if not ("float" in str(wav_data.dtype)):
        # scale to -1.0 -- 1.0
        if wav_data.dtype == 'int16':
            nb_bits = 16  # -> 16-bit wav files
        elif wav_data.dtype == 'int32':
            nb_bits = 32  # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        wav_data = wav_data / (max_nb_bit + 1)  # samples is a numpy array of floats representing the samples 
    if return_df:
        # Convert to dataframe:
        signal_df = pd.DataFrame(columns=["time", "signal"])
        signal_df['signal'] = wav_data
        signal_df['time'] = signal_df.index * (1 / fs)
        return signal_df, fs
    else:
        return wav_data, fs


# =============================================================================
# Framing pandas Series:
def stride_trick(signal, frame_len, frame_step, n_frames):
    """
     apply framing using the stride trick from numpy.

     Args:
         signal (array) : signal array.
         frame_len (int) : length of the stride.
         frame_step (int) : stride step.
         n_frames: number of frames
     Returns:
         framed array.
     """

    n = signal.strides[0]
    return np.lib.stride_tricks.as_strided(signal,
                                           shape=(n_frames, frame_len),
                                           strides=(frame_step * n, n))


def framing(signal, fs=16000, win_len=0.05, win_hop=0.01):
    """
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://superkogito.github.io/blog/SignalFraming.html
    :param signal: pandas Series (N,)
    :param fs: in Hz
    :param win_len:in sec
    :param win_hop:in sec
    :return: 2d array of [frames x  frame length]
    """

    ### Option 2: 
    # https://superkogito.github.io/blog/2020/01/25/signal_framing.html
    # tic_toc.tic()
    if win_len < win_hop: print("ParameterError: win_len must be larger than win_hop.")

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = np.round(win_len * fs).astype(int)
    frame_step = np.round(win_hop * fs).astype(int)
    signal_length = len(signal)
    frames_overlap = frame_length - frame_step

    # compute number of frames and left sample in order to pad if needed to make
    # sure all frames have equal number of samples  without truncating any samples
    # from the original signal
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(signal, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

    n_frames = ((signal_length - frame_length) // frame_step) + 1
    if n_frames < 0:
        print("stride_trick: n_frames is negative")
    # apply stride trick
    frames = stride_trick(pad_signal, frame_length, frame_step, n_frames)
    # print("framing option 2: {:.5f}".format(tic_toc.toc()))
    return frames


# =============================================================================
def energy_sig_framed(framed_sig, db=False):
    """
    Calculate energy for each frame in the framed signal.
    :param framed_sig: 2d array, [frames x frame length]
    :param db: True: convert energy to decibels. False: do not convert.
    :return: energy for each frame. array.
    """
    energy = (np.sum(framed_sig **2, axis=1) ) / framed_sig.shape[1] # Fixed on 08.12.24.
    if db:
        energy = 10 * np.log10(energy)
    return energy


# calculate energy of for each frame of a signal
# =============================================================================
def calc_seg_energy(signal, fs, frame_length, frame_step):
    """
    :param signal: a Series vector
    :param fs: sampling frequency [Hz]
    :param frame_length: frame size [sec]
    :param frame_step: step size in framing method [sec]
    :return: energy for each frame. array.
    """
    # Frame the segment:
    frames = framing(signal, fs, win_len=frame_length, win_hop=frame_step)
    # Calculate energy of the segment:
    return energy_sig_framed(frames)  # [energy_frame(frame) for frame in frames]


# unite adjacent labels of the same speaker:
# =============================================================================
def unite_labels(annotations):
    """
    the function unite sequential sequential labels:
    Example: annotations=
    t1 t2 speaker1
    t2 t3 speaker1
    t3 t4 speaker1
    Will return: new_annotations =
    t1 t4 speaker1

    :param annotations: Dataframe with 3 columns: start, end, speaker
    :return: new_annotations: Dataframe with 3 columns: start, end, speaker, with united labels
    """
    temp_start = annotations['start'][0]
    temp_speaker = annotations['speaker'][0]
    new_annotations = []
    for i in np.arange(1, annotations.shape[0]):
        if (temp_speaker == annotations['speaker'][i]) & \
                (annotations['end'][i - 1] == annotations['start'][i]):
            continue
        else:
            new_annotations.append({'start': temp_start,
                                    'end': annotations['end'][i - 1],
                                    'speaker': temp_speaker})
            temp_speaker = annotations['speaker'][i]
            temp_start = annotations['start'][i]

    new_annotations.append({'start': temp_start,
                            'end': annotations['end'][i],
                            'speaker': temp_speaker})
    return pd.DataFrame(new_annotations)


# Round number half up: 2.5->3, 2.3-> 2.
# =============================================================================
def round_to_int(num, d=0):
    """
    12.09.2022
    :type d: integer
    """

    mult_num = num * 10 ** d
    num_sign = int(np.sign(num))
    # Check the remaining number after the decimal point
    num_after_deci = mult_num - np.floor(mult_num)
    if num_after_deci >= 0.5:
        if d == 0:  # 27.09.2023. integer
            return num_sign * int(int(np.floor(np.abs(mult_num)) + 1) / 10 ** d)
        else:  # 27.09.2023. if want to return not integer but float number with less numbers after decimal point
            return num_sign * int(np.floor(np.abs(mult_num)) + 1) / 10 ** d
    else:
        if d == 0:  # 27.09.2023. integer
            return num_sign * int(np.floor(np.abs(mult_num)) / 10 ** d)
        else:  # 27.09.2023. if want to return not integer but float number with less numbers after decimal point
            return num_sign * np.floor(np.abs(mult_num)) / 10 ** d


# Get number of digits after the decimal point
# =================================================================================================
def get_num_digits_after_point(num):
    return len(str(num)) - 2  # 0.11->2. 0.110-> 2.


# Keep only N random Speech events for each recording to keep events balance
# =================================================================================================
def keep_N_random_events(data_dict, n_p_keep=10, class_label=2, df_vocs=None):
    import zlib
    # If keep ALL events do nothing and return the same dict:
    if n_p_keep == 100:
        if isinstance(df_vocs, pd.DataFrame) and not df_vocs.empty:
            return data_dict, df_vocs
        else:
            return data_dict

    data_dict_reduced = copy.deepcopy(data_dict)
    recs = np.unique(data_dict["group"])
    selected_indices = []

    for grp in recs:
        # Generate seed for this spesific recording:
        grp_str = str(grp).encode() # Convert to bytes
        seed = zlib.crc32(grp_str)  # Generate a stable integer hash
        random.seed(seed) # instead of: int(grp))
        # Filter samples by group and target class
        group_indices = np.where(data_dict["group"] == grp)[0]
        class_indices = [idx for idx in group_indices if data_dict["y"][idx] == class_label]

        # Count number of Speech events in a recording:
        num_speech_events = sum((data_dict["y"] == class_label) & (data_dict["group"] == grp))
        n_keep = int(num_speech_events * n_p_keep / 100)  # Number of Speech events to keep

        # Randomly select K samples from the filtered indices
        if len(class_indices) > n_keep:
            selected_indices.extend(np.sort(random.sample(class_indices, n_keep)))
        else:
            selected_indices.extend(class_indices)

        non_target_class_indices = [idx for idx in group_indices if idx not in class_indices]
        selected_indices.extend(non_target_class_indices)

    # Create new filtered arrays
    for var in data_dict_reduced:
        data_dict_reduced[var] = data_dict[var][np.sort(selected_indices)]
    
    print(f"Number of samples in Train after reducing Speech samples: {np.bincount(data_dict_reduced['y'])}")
    
    if isinstance(df_vocs, pd.DataFrame) and not df_vocs.empty:
        df_vocs = df_vocs.iloc[selected_indices].reset_index(drop=True)
        return data_dict_reduced, df_vocs

    return data_dict_reduced

# -------------------------------------------------------------------------------------------------
def pad_trim_audio(vs, fs, target_len=None, min_len=0, if_print=True):
    '''
    Trim the vocal segment to multiple segments with size "target_len" (pad the short ones)
    OR pad the short vocal segment to size "target_len"
    Parameters
    ----------
    vs : numpy array
        One vocal segment.

    Returns
    -------
    vs_sliced : list of numpy arrays
        Each array in the list is a sliced vocal segment to the "seg_len" size
    '''
    if not target_len:  # if keep the original length of the vocalization
        return [vs]

    # Else: pad+trim
    vs_sliced = list()

    if len(vs) > target_len * fs:  # trim the signal to seg_len segments
        n_segs = int(np.ceil(len(vs) / fs / target_len))  # number of segments can be created with size "seg_len"
        start_flag = 0
        for i in range(n_segs):
            if i != (n_segs - 1):  # trim the segment to size "seg_len"
                vs_sliced.append(vs[start_flag: int(target_len * fs * (i + 1))])
                start_flag = int(target_len * fs * (i + 1))
            else:  # pad the last /only segment to size "seg_len"
                # if the length of the resulted vs is longer than the minimum duration:
                # otherwise don't pad
                if len(vs[start_flag:]) >= int(np.ceil(min_len * fs)):  # 08.08.22.
                    vs_sliced.append(np.append(vs[start_flag:],
                                               np.zeros(int(target_len * fs) - len(
                                                   vs[start_flag:]), )))
                else:
                    if if_print:
                        print(
                            "Trimmed VS is ignored because with len = {:.3f}sec < {:.3f}sec".format(
                                len(vs[start_flag:]) / fs, min_len))
    else:  # pad the signal to seg_len
        vs_sliced = [np.append(vs, np.zeros((int(target_len * fs) - len(vs),)))]
    return vs_sliced