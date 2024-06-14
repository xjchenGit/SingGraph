import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from utils import str_to_bool

from RawBoost import process_Rawboost_feature
from BPMProcess import BpmProcessor

___author__ = "Xuanjun Chen"
__email__ = "d12942018@ntu.edu.tw"

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def pad_random(x: np.ndarray, start: int, end: int):
    x_len = x.shape[0]

    # If the interval is within the length of x
    if end <= x_len:
        return x[start:end]

    # If the selected interval is longer than x
    padded_x = np.tile(x, (end // x_len + 1))  # Repeat x to ensure it covers the interval
    return padded_x[start:end]

class Dataset_SingFake(Dataset):
    def __init__(self, args, base_dir, algo, state, is_mixture=False, target_sr=16000):
        """
        base_dir should contain mixtures/ and vocals/ folders
        """
        self.base_dir = base_dir
        self.is_mixture = is_mixture
        self.target_sr = target_sr
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.args = args
        self.algo = algo
        self.state = state
        
        # get file list
        self.file_list = []
        if self.is_mixture:
            self.target_path = os.path.join(self.base_dir, "mixtures")
        else:
            self.target_path = os.path.join(self.base_dir, "vocals")
            
        print(self.target_path)
        
        assert os.path.exists(self.target_path), f"{self.target_path} does not exist!"
        
        for file in os.listdir(self.target_path):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        file_path = os.path.join(self.target_path, key + ".flac")
        # X, _ = sf.read(file_path, samplerate=self.target_sr)
        try:
            X, fs = librosa.load(file_path, sr=self.target_sr, mono=False)
        except:
            return self.__getitem__(np.random.randint(len(self.file_list)))
        if X.shape[0] > 1:
            # if not mono, take random channel
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]
        
        # RawBoost Augmentation
        if self.state == "train":
            X = process_Rawboost_feature(X, fs, self.args, self.algo)
            
        X_pad = pad_random(X, self.cut)
        X_pad = X_pad / np.max(np.abs(X_pad))
        x_inp = Tensor(X_pad)
        y = int(key.split("_")[0])
        return x_inp, y

class Dataset_SingFake_mert_w2v(Dataset):
    def __init__(self, args, config, base_dir, algo, state,
                 target_sr=16000, target_sr2=24000):
        """
        base_dir should contain mixtures/ and vocals/ folders
        """
        self.base_dir = base_dir
        self.is_mixture = not str_to_bool(config["vocals_only"])
        self.is_sep = str_to_bool(config["is_sep"])
        self.is_rawboost = str_to_bool(config["is_rawboost"])
        self.is_beat_matching = str_to_bool(config["is_beat_matching"])
        
        self.target_sr = target_sr
        self.target_sr2 = target_sr2
        self.cut16 = 64600  # take ~4 sec audio (64600 samples)
        self.duration = 4.0375
        self.cut24 = 96900  # take ~4 sec audio (96900 samples)
        self.args = args
        self.algo = algo
        self.state = state
        
        # get file list
        self.file_list = []
        if self.is_mixture:
            self.target_path = os.path.join(self.base_dir, "mixtures")
            self.tgt_v2_path = self.target_path
        elif not self.is_mixture and self.is_sep:
            self.target_path = os.path.join(self.base_dir, "vocals")
            self.tgt_v2_path = os.path.join(self.base_dir, "non_vocals")
        else:    
            self.target_path = os.path.join(self.base_dir, "vocals")
            self.tgt_v2_path = self.tgt_v2_path
            
        self.beat_file_path = os.path.join(self.base_dir, "beats")
        
        # For beat matching
        self.BpmProCls = BpmProcessor(train_acc_path=config["train_acc_path"],
                                      json2bpm_path=config["j2b_path"],
                                      bpm2json_path=config["b2j_path"],
                                      sample_rate=16000,
                                      threshold=self.cut16)
        
        assert os.path.exists(self.target_path), f"{self.target_path} does not exist!"
        
        for file in os.listdir(self.target_path):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])
        
        # Filter 0 or 1
        
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        y = int(key.split("_")[0])

        file_path = os.path.join(self.target_path, key + ".flac")
        file2_path = os.path.join(self.tgt_v2_path, key + ".wav")
        try:
            X, fs = librosa.load(file_path, sr=self.target_sr, mono=False)
            X2, _ = librosa.load(file2_path, sr=self.target_sr, mono=False)
        except:
            return self.__getitem__(np.random.randint(len(self.file_list)))

        if X.shape[0] > 1 or X2.shape[0] > 1:
            # If not mono, take random channel
            channel_id = np.random.randint(X.shape[0])
            X, X2 = X[channel_id], X2[channel_id]
            
        if self.state == "train" and self.is_beat_matching:
            bpm_n = self.BpmProCls.j2b_dict[key + ".json"]["bpm"]
            db_num = len(self.BpmProCls.j2b_dict[key + ".json"]["downbeats"])
            if bpm_n is not None and db_num > 2:
                X = self.BpmProCls.sv_beat_align(X, key)
                sel_json, dbs_duration = self.BpmProCls.sel_accom_from_bpm_group(bpm_n, y)
                if dbs_duration != 0:
                    sel_wav = self.BpmProCls.load_audio_by_json(sel_json)
                    X2 = self.BpmProCls.accom_beat_padding(sel_wav, sel_json, dbs_duration)

        # RawBoost Augmentation
        if self.state == "train" and self.is_rawboost:
            X = process_Rawboost_feature(X, fs, self.args, self.algo)

        waveform_shift = X.shape[0] - self.cut16
        if waveform_shift > 0:
            x_start = np.random.randint(0, waveform_shift)
        else:
            x_start = 0
        
        x_end = x_start + self.cut16
        X_pad, X2_pad = pad_random(X, x_start, x_end), pad_random(X2, x_start, x_end)
        X_pad, X2_pad = X_pad / np.max(np.abs(X_pad)), X2_pad / np.max(np.abs(X2_pad))
        
        x_inp, x2_inp = Tensor(X_pad), Tensor(X2_pad)
        
        
        return x_inp, x2_inp, y
