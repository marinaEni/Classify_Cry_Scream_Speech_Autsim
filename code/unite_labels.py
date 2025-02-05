import pandas as pd
import numpy as np
import os
from pathlib import Path

from common import get_database_path


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
    if annotations.shape[0] == 1:
        return annotations
    temp_start = annotations['start'][0]
    temp_speaker = annotations['speaker'][0]
    temp_event = annotations['event'][0]
    new_annotations = []
    for i in np.arange(1, annotations.shape[0]):
        if (temp_speaker == annotations['speaker'][i]) & \
                (annotations['end'][i - 1] == annotations['start'][i]):
            continue
        else:
            new_annotations.append({'start': temp_start,
                                    'end': annotations['end'][i - 1],
                                    'speaker': temp_speaker,
                                    'event': temp_event})
            temp_speaker = annotations['speaker'][i]
            temp_start = annotations['start'][i]
            temp_event = annotations['event'][i]

    new_annotations.append({'start': temp_start,
                            'end': annotations['end'][i],
                            'speaker': temp_speaker,
                            'event': temp_event})
    return pd.DataFrame(new_annotations)


def main():

    man_seg_path = get_database_path('man_seg')
    man_seg_united_path = get_database_path('man_seg_united')
    rec_list = [f for f in os.listdir(man_seg_path) if f.endswith('.txt')]
    
    for rec_id in rec_list:
        
        rec_id_save_path = Path(man_seg_united_path) / Path(rec_id) # <path>/<rec_id>.txt
        # # check if united labels exist for this rec_id:
        # if os.path.isfile(rec_id_save_path):
        #     print(rec_id, "exist")
        #     continue
        man_seg = pd.read_csv(Path(man_seg_path) / Path(rec_id), delim_whitespace=True, header=None)
        man_seg.columns = ["start", "end", "speaker", "event"]
        man_seg_united = unite_labels(man_seg)
        # Save the united labels to txt file:
        man_seg_united.to_csv(rec_id_save_path, sep='\t', #float_format='%.2f', # 06.09.22: no float format
                              header=False, index=False)
        print(rec_id)
    print('Done all.')

# =============================================================================
if __name__ == "__main__":
    main()