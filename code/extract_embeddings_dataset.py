# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:16:27 2023

@author: marinamu
"""

from common import get_database_path, round_to_int, load_wav, pad_trim_audio
from run_trained_yamnet_mdl import RunYAMNET


from tqdm import tqdm
from pathlib import Path
import numpy as np
from transformers import Wav2Vec2Model, WavLMForXVector, Wav2Vec2Processor 
import torch
import pandas as pd
import os
import pickle
import openl3  # pip install openl3
from whisper import load_model  # pip install openai-whisper
from whisper.audio import (
    N_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim)


class ExtractEmbeddings:

    def __init__(self, data_df, config, codebook_name, should_print=True):
        self.data_df = data_df
        self.config = config
        self.embeddings_config = config["embeddings_config"]
        self.codebook_name = codebook_name
        self.should_print = should_print

        self.seg_len = self.embeddings_config.get('seg_len', None)  # in sec.
        self.min_seg_len = self.embeddings_config.get('min_seg_len')  # in sec

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.define_feat_gen()
        self.recs_path = None # path where are the recordings
        
        # List of pre-trained models that work on each vocalziation seperately:
        self.list_models_per_voc = ["yamnet", "eGeMAPSv02", "whisper-base", "whisper-small", "whisper-tiny", "whisper-medium", "wav2vec2-base-960h", "wav2vec2-base"]
        if not self.seg_len: # if not pad/trim then calculate wav2vec features for each vocalization seperately
            self.list_models_per_voc.append("wav2vec2-base-960h")
            
    # =============================================================================================
    def print_if_needed(self, message):
        if self.should_print:
            print(message)
    
    @staticmethod
    def extract_wavlm_embeddings(model, feature_extractor, data, fs, device):
        """Use WavLM model to extract embeddings for audio segments"""
        emb = list()
        for i in tqdm(range(len(data))):
            inputs = feature_extractor(
                data[i], 
                sampling_rate=fs, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            with torch.no_grad():
                embeddings = model(**inputs).embeddings

            emb += torch.nn.functional.normalize(embeddings.cpu(), dim=-1).cpu()

        return torch.stack(emb)
    
    def extract_vggish_embeddings(self, data, fs, target_length = 0.96):
        # The audio must be at least 0.975 seconds long.
        # frame_length + (n_min_frames - 1) * hop_length # 15600 samples == 0.975seconds
        import vggish_slim
        import vggish_params
        import vggish_input
        import tensorflow.compat.v1 as tf
        """
        git clone https://github.com/tensorflow/models.git
        cd models/research/audioset/vggish
        pip install -r requirements.txt
        """
        def get_vggish_emb(audio_signals, sample_rate, model_checkpoint='vggish_model.ckpt'):
            """
            Extract VGGish embeddings from audio signals.
            
            Parameters:
                audio_signals (list of np.ndarray): List of audio signals (1D numpy arrays).
                sample_rate (int): Sampling rate of the audio signals.
                model_checkpoint (str): Path to the VGGish model checkpoint.
                
            Returns:
                list of np.ndarray: Embeddings for each audio signal.
            """
            # Disable TensorFlow eager execution for compatibility with VGGish
            tf.compat.v1.disable_eager_execution()

            # Load the VGGish model
            tf.reset_default_graph()
            with tf.Session() as sess:
                vggish_slim.define_vggish_slim()
                vggish_slim.load_vggish_slim_checkpoint(sess, model_checkpoint)
                
                # TensorFlow placeholders for input and output
                features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                
                embeddings = []

                for audio in audio_signals:
                    # Convert waveform to examples (log mel spectrogram patches)
                    examples = vggish_input.waveform_to_examples(audio, sample_rate)
                    print(f"Audio shape: {audio.shape}, Examples shape: {examples.shape}")
                    
                    # Run the model and get embeddings
                    [embedding] = sess.run([embedding_tensor], feed_dict={features_tensor: examples})
                    embeddings.append(embedding)
                    print(f"Extracted embeddings shape: {embedding.shape}")
            
            return embeddings
        
        # Define frame length (400 samples) and hop length (160 samples)
        frame_length = int(0.025*fs)
        hop_length = int(0.010*fs)
        n_min_frames = int(target_length / 0.01) # 96 frames
        
        # Calculate the minimum required length for the audio to accommodate the frame and hop size
        min_samples = frame_length + (n_min_frames - 1) * hop_length # 15600 samples == 0.975seconds
        print(f"Required minimum audio length: {min_samples} samples")
        audio_padded_files = []
        embeddings = []
        for arr in data:
            print(f"arr shape {arr.shape}")
            # Pad to min_samples at the end if needed:
            if len(arr) < min_samples:
                # Calculate how much padding is needed
                padding_needed = min_samples - len(arr)
                print("padding needed", padding_needed)
                # Pad with zeros (or other value if you prefer)
                audio_padded = np.pad(arr, (0, int(padding_needed)), mode='constant', constant_values=0)
                print(f"audio_padded shape {audio_padded.shape}")
                audio_padded_files.append(audio_padded)
            else:
                audio_padded_files.append(arr)
                print("NO padding needed")

        embeddings = get_vggish_emb(audio_padded_files, sample_rate=fs) # list of arrays
        return embeddings
    # =============================================================================================
    def define_feat_gen(self):
        print("define_feat_gen")
        if self.embeddings_config["model_extractor"] == "yamnet":
            # Load the model only once for time saving:
            self.yam_cls = RunYAMNET(should_print=self.should_print)
            self.yam_cls.load_model_labels()
            
        elif "wav2vec" in self.embeddings_config["model_extractor"]:
            # Load the wav2vec2 model
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                "facebook/{}".format(self.embeddings_config["model_extractor"])).to(self.device)  # "wav2vec2-base-960h"

        elif self.embeddings_config["model_extractor"] == 'openl3':
            self.openl3_model = openl3.models.load_audio_embedding_model(
                input_repr=self.embeddings_config["input_repr"],
                content_type=self.embeddings_config["content_type"],
                embedding_size=self.embeddings_config["embedding_size"])
            # Print the model summary to understand its structure
            # self.openl3_model.summary()
            
        elif "whisper" in self.embeddings_config["model_extractor"]:
            # Model
            print("Loading whisper model..")
            model_name = self.embeddings_config["model_extractor"].split("-")[1] + ".en"
            self.whisper_decode_model = load_model(model_name).eval()
            print("Whisper Model Loaded!")

        elif "eGeMAPSv02" in self.embeddings_config["model_extractor"]:
            import opensmile
            # Define the feature set:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet["eGeMAPSv02"],
                feature_level=opensmile.FeatureLevel.Functionals)
        print("Done")

    # Run embedding extraction for each event for every recording in the data:
    # =============================================================================================
    def generate_embeddings(self):
        # Get list of recordings names from the dataframe:
        recs_data = self.get_recs_list()

        self.X, self.y, self.y_pred, self.group, self.pred_scores = dict(), dict(), dict(), dict(), dict()

        # Run on every recording:
        for i_rec, rec_id in enumerate(recs_data):
            # Load recording:
            wav_data, fs = load_wav(rec_id, self.recs_path, return_df=False)
            # Get vocalizations dataframe (Start, end, event) of the recording:
            vocal_segments_rec = self.data_df[self.data_df.Recording.isin([rec_id])]
            if vocal_segments_rec.shape[0] == 0:  # empty = the child doesnt have events
                continue
            self.X[rec_id], self.y[rec_id], self.y_pred[rec_id], self.group[
                rec_id], pred_scores = [], [], [], [], []

            if not self.exists(rec_id) or self.embeddings_config["model_extractor"] == "yamnet":
                vocs_rec = []
                for _, vs_df in vocal_segments_rec.iterrows():
                    # Take single vocal segment:
                    i_start = round_to_int(vs_df['start'] * fs,
                                           d=0)  # d = means return integers (no values after point)
                    i_end = round_to_int(vs_df['end'] * fs, d=0)
                    vs = wav_data[i_start:i_end - 1]
                    # Pad it to the seg_len:
                    vs_padded = pad_trim_audio(
                        vs, fs, target_len=self.seg_len, min_len=self.min_seg_len) # output: list
                    
                    # For each sliced vocalization valculate embeddings:
                    for vs in vs_padded:
                        self.y[rec_id].append(vs_df["event"])
                        self.group[rec_id].append(rec_id)
                        if self.embeddings_config["model_extractor"] in self.list_models_per_voc:
                            # vocs_rec.append(vs)  # 24.12.23. instead of extracting from each voc, extract from list of vocs. 
                            feats_voc, y_pred, pred_score = self.extract_embeddings(vs, fs)
                            # Extract embeddings:
                            self.X[rec_id].append(feats_voc)
                            # Predict class (Yamnet):
                            self.y_pred[rec_id].append(y_pred)
                            pred_scores.append(pred_score)  # 04.10.2023.
                        else:
                            vocs_rec.append(vs)

                if not self.embeddings_config["model_extractor"] in self.list_models_per_voc:
                    self.X[rec_id], _, _ = self.extract_embeddings(vocs_rec, fs)

                elif self.embeddings_config["model_extractor"] == "yamnet":
                    # for each recording Create a dataframe of predicted score for each class:
                    self.pred_scores[rec_id] = self.create_pred_scores_df(pred_scores)

                if self.config["save_embeddings"]:  # if to save to file
                    self.save_embeddings_rec(rec_id, self.X[rec_id], self.y[rec_id])

            else:  # Load embeddings:
                self.X[rec_id], self.y[rec_id] = self.load_embeddings_rec(rec_id)
                self.group[rec_id] = [rec_id] * len(self.y[rec_id])

            self.print_if_needed(
                "Done extracting embeddings for {}/{}".format(i_rec + 1, len(recs_data)))

        return self.X, self.y, self.y_pred, self.group

    # Convert list of predicted score to a dataframe with columns as YAMNTet class names:
    # =============================================================================================
    def create_pred_scores_df(self, score_per_voc_list):
        records = []
        for voc_i, score_array in enumerate(score_per_voc_list):
            record = {'Vocalization': voc_i}
            record.update({class_name: score
                           for score, class_name in
                           zip(np.squeeze(score_array), self.yam_cls.class_names)})
            records.append(record)

        return pd.DataFrame(records)

    # Get list of recordings names from the dataframe:
    # =============================================================================================        
    def get_recs_list(self):
        return self.data_df["Recording"].unique().tolist()

    # Padd a numpy array with pad_val value in random position (beginning or/end):
    # =============================================================================================
    def random_pad_array(self, inner_array, in_target_size, pad_val=0):
        """
        Perform random padding of an array with a specific target size.
    
        Args:
            inner_array (numpy.ndarray): The input array to be padded.
            in_target_size (int): The desired size of the padded array.
            pad_val (int or float, optional): The value used for padding. Default is 0.
    
        Returns:
            numpy.ndarray: The padded array.
        """

        current_size = inner_array.shape[0]
        padding_size = in_target_size - current_size

        # Generate a random number to determine the amount of padding at the beginning
        pad_before = np.random.randint(0, padding_size + 1)
        pad_after = padding_size - pad_before

        # Pad the array with zeros using numpy.pad
        padded_array = np.pad(inner_array, ((pad_before, pad_after),), mode='constant')
        return padded_array

    # =============================================================================================
    def extract_embeddings(self, wav_data, fs):
        pred_class_name, pred_scores = None, None

        if self.embeddings_config["model_extractor"] == "yamnet":
            # Extract embeddings from yhe wav_data:
            self.yam_cls.waveform = wav_data
            self.yam_cls.original_fs = fs
            self.yam_cls.prepare_audio()
            self.yam_cls.predict_class()
            features = np.array(self.yam_cls.embeddings)  # (n_chunks, 1024). array of num_chunks of 0.975sec X 1024
            pred_class_name = self.yam_cls.get_out_class()
            pred_scores = self.yam_cls.scores_np
            print(f"features shape before pooling: {features.shape}")
            features = self.apply_pooling(features)  # output: (1024,)
            print(f"features shape after pooling: {features.shape}")
            
        elif "wav2vec" in self.embeddings_config["model_extractor"]:
            
            if len(wav_data.shape) == 1:
                wav_data = np.expand_dims(wav_data,axis=0)
                
            # Convert the preprocessed audio to a tensor and move it to the device
            inputs = torch.from_numpy(np.array(wav_data)).float().to(self.device)
            print(f"inputs shape: {inputs.shape}")

            # Extract features from the audio
            with torch.no_grad():
                outputs = self.wav2vec_model(inputs)

            # Access the extracted features
            features = outputs.last_hidden_state.cpu().numpy() # (1,X,768)
            # print(f"features last_hidden_state: \n{features}")
            print(f"features shape: {features.shape}")
            features = self.apply_pooling(np.squeeze(features), axis=0) # (768,)
            print(f"features shape after pooling: {features.shape}")
            
        # -----------------------------------------------------------------------------------------
        elif 'openl3' in self.embeddings_config["model_extractor"]:
            features, ts1 = openl3.get_audio_embedding(wav_data, fs,
                                                       model=self.openl3_model,
                                                       center=self.embeddings_config["center"],
                                                       # if center=False -> output=(1,512), otherwise: (6,512)
                                                       verbose=0)  # default hop_size=0.1s. output: (6,512)
            print(f"Openl3 features shape before pooling: {features[0].shape}")
            features = self.apply_pooling(features)  # output: (512,)
            print(f"Openl3 features shape after pooling: {features[0].shape}")
        # -----------------------------------------------------------------------------------------
        elif "whisper" in self.embeddings_config["model_extractor"]:
            # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/whisper/feature_extraction_whisper.py#L32
            # Convert the preprocessed audio to a tensor and move it to the device
            input_data = torch.tensor(np.array(wav_data), dtype=torch.float32).to(self.device)
            # Pad 30-seconds of silence to the input audio, for slicing
            audio = pad_or_trim(input_data.flatten())
            mel = log_mel_spectrogram(audio).cpu().numpy().astype(np.float32)  # (80, 3000)
            content_frames = mel.shape[-1]  # number of frames (3000)

            dtype = torch.float32
            seek = 0
            while seek < content_frames:
                mel_segment = torch.Tensor(mel[:, seek: seek + N_FRAMES]).to(
                    self.whisper_decode_model.device).to(dtype)
                segment_size = min(N_FRAMES, content_frames - seek)  # 3000

                mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(
                    self.whisper_decode_model.device).to(dtype)  # (80,3000)
                encoder_out = self.whisper_decode_model.encoder(
                    mel_segment.unsqueeze(0))  # encoder forward pass (AudioEncoder class)
                seek += segment_size
                print("encoder_out shape: ", encoder_out.shape)
                # to discard the extra padding embeddings (if audio < 30 sec):
                encoder_out_trim = encoder_out[:, :50 * len(input_data) // SAMPLE_RATE,
                                   :].squeeze().detach().cpu().numpy()  # embeddings rate = 50Hz (50 samples per 1sec,512)
                print(f"features shape of voc with shape {len(input_data)} before pooling: {encoder_out_trim.shape}")
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(encoder_out_trim) 
                plt.show()
                features = self.apply_pooling(encoder_out_trim)  # (,512)
                print(f"features shape after pooling: {features.shape}")
                # print(f"features shape: {features.shape}")
                break
            
        elif "vggish" in self.embeddings_config["model_extractor"]:            
            embeddings = self.extract_vggish_embeddings(wav_data, fs)
            print("features shape before pooling", [len(emb) for emb in embeddings])
            features = self.apply_pooling(embeddings)  # (,128)
            print("features shape after pooling", [len(emb) for emb in features])
            
        elif "eGeMAPSv02" in self.embeddings_config["model_extractor"]:
            # Extract features:
            features = np.squeeze(
                self.smile.process_signal(wav_data, fs).to_numpy())  # 1x88 -> (88,)
            
        return features, pred_class_name, pred_scores

    # Apply pooling on embeddings rows of a vocalization to create embeddings vector:
    # =============================================================================================
    def pooling_embeddings(self, embeddings, axis=0):
        # If only one vector do nothing:
        if embeddings.shape[0]==1:
            return np.squeeze(embeddings)
        
        # Mask NaN values
        masked_embeddings = np.ma.masked_where(np.isnan(embeddings), embeddings) # (embeddings == 0) | 

        if (self.embeddings_config.get("feat_pool") is None) or (
                not self.embeddings_config.get("feat_pool")):  # if doesnt exist or if empty
            self.embeddings_config["feat_pool"] = "max"
        
        if self.embeddings_config["feat_pool"] == "avg":
            # Calculate row-wise mean, ignoring NaNs
            return masked_embeddings.mean(axis=axis).filled(0)  # Fill with 0 if all values are masked
        elif self.embeddings_config["feat_pool"] == "max":
            return masked_embeddings.max(axis=axis).filled(0)  # Fill with 0 if all values are masked
        elif self.embeddings_config["feat_pool"] == "min":
            return masked_embeddings.min(axis=axis).filled(0)  # Fill with 0 if all values are masked
        elif self.embeddings_config["feat_pool"] == "sum":
            return masked_embeddings.sum(axis=axis).filled(0)  # Fill with 0 if all values are masked
        # np.nansum(embeddings, axis=axis)

    # =============================================================================================
    def apply_pooling(self, embeddings, axis=0):
        if isinstance(embeddings, list):
            new_emb = []
            for emb in embeddings:
                new_emb.append(self.pooling_embeddings(emb, axis))
        else:
            new_emb = self.pooling_embeddings(embeddings, axis)
        return new_emb

    # =============================================================================================
    def exists(self, rec_id):  # check if the file of the embeddings of a rec already exist
        return os.path.isfile(self.get_file_path(rec_id)) and (not self.config["generate"])

    # =============================================================================================
    def get_file_path(self,
                      rec_id) -> str:  # return the rec's embeddings location file full path name
        return self.get_dir_path() / Path('{}.p'.format(rec_id))

    # =============================================================================================
    def get_dir_path(self):  # get the path theat fits the configuration to the embeddings
        return get_database_path('embeddings') / Path(self.codebook_name)

    # =============================================================================================
    def save_embeddings_rec(self, rec_id, embeddings_rec,
                            y_true):  # save embeddings of a rec to a file
        """
        The function saves the embeddings as a list to a pickle file.
        """
        path = self.get_dir_path()
        if not os.path.isdir(path):
            os.makedirs(path)
        parameters_dict = {"embeddings_rec": embeddings_rec,
                           "y_true": y_true}
        # Save to pickle:
        pickle.dump(parameters_dict, open(self.get_file_path(rec_id), "wb"))
        print(f'Saved embeddings in {path}')

    # =============================================================================================
    def load_embeddings_rec(self, rec_id):  # Load embeddings of a rec
        loaded_parameters = pickle.load(open(self.get_file_path(rec_id), "rb"))
        embeddings_rec = loaded_parameters["embeddings_rec"].copy()
        y_true = loaded_parameters["y_true"].copy()
        print('Loaded embeddings')
        return embeddings_rec, y_true
