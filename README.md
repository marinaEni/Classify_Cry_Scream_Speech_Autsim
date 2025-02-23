# Classify_Cry_Scream_Speech_Autsim
Classify audio events to cry&amp;scream vs speech

This algorithm classifies audio events from speech recordings into cry&scream or speech using transfer learning. The algorithm is based on features/embeddings extracted from YAMNet, VGGish, Whisper, Wav2Vec2, OpenL3, and eGeMAPs. The classifier is a multilayer perceptron with two fully connected hidden layers.

## Folders organization
•	`./code`: python files used to train and test the model (`.code/main_event_classify.py`) using 5-fold cross-validation.

•	`./config`: the configuration files: `./config/config_trans_learn.yaml`.

•	`./data`: sample data that includes:
1. `./manual_annotations`: Excel files, one file per recording. Each file includes 4 columns: event start time, event end time, speaker, and event class. The names of the Excel files must match the recording's names.
2. `./recs_lists`: includes .yaml file/s of list of recordings used in the process: `recs_list_filename.yaml`.
3. `./recordings`: .wav files (16kHz) of the recordings. 

•	`./codebooks`: include a txt file with the features configurations codebook. Each new configuration receives a name: "config_<date today>_<time now>"

•	`./results`: includes:
1. `./models`: the trained models for each fold. For example:
   ![image](https://github.com/user-attachments/assets/81066222-4f51-4d84-b57a-4e77c6cf5bef)

   Each fold includes:
   
   ![image](https://github.com/user-attachments/assets/2f1d5de7-883d-4a24-9f39-75124889c840)

3. `./embeddings`: the features/embeddings as pickle files for each configuration name. The features of the same configuration names are saved in a specific folder.
   For example:
   
   ![image](https://github.com/user-attachments/assets/8908e369-3c73-429c-aa67-37c2689c60e4)


## Training

This process includes 5-fold cross-validation, where each fold includes hyper-parameters tuning (learning rate, decay rate, batch size, and number of epochs), testing the best parameters on the fifth fold, and performance evaluation.
If the user chooses to apply under-sampling (removing random Speech sample to x%), then the main run folder will be generated a folder with the results: `UnderSample<x>.`
If the user chooses to apply over-sampling (adding synthetic samples (SMOTE) to the minority class), then the main run folder will be generated a folder with the results: `SMOTE.`


# Run
To make it run properly, clone this repository in a folder.
From your command line, go to `ClassifyCryScreamSpeech/code` folder and run the following Python scripts:

``` python
# Run training using the configuration file
python main_event_classify.py -c ../config/config_trans_learn.yaml
```

