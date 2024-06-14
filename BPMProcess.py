import os, math, json
import numpy as np
import librosa

class BpmProcessor:
    def __init__(self, train_acc_path,
                 json2bpm_path,
                 bpm2json_path,
                 sample_rate,
                 threshold):
        with open(json2bpm_path, "r") as f:
            j2b_dict = json.load(f)
        
        with open(bpm2json_path, "r") as f:
            b2j_dict = json.load(f)
        
        self.train_acc_path = train_acc_path
        self.j2b_dict = j2b_dict
        self.b2j_dict = b2j_dict
        
        self.sr = sample_rate
        self.thr = threshold / self.sr
        self.waveform_thr = threshold ## 1000000
    
    def load_audio_by_json(self, sel_json_name):
        sel_wav_path = os.path.join(self.train_acc_path, sel_json_name[:-5] + ".wav")
        wav, _ = librosa.load(sel_wav_path, sr=self.sr)
        return wav
        
    def sel_accom_from_bpm_group(self, bpm_num, y):
        count, downbeats_duration = 0, 0
        sel_json = None
        candidate_jsons = self.b2j_dict[str(bpm_num)]
        filtered_gen = filter(lambda x: x.startswith(str(y)), candidate_jsons)
        candidate_list = list(filtered_gen)
        
        if len(candidate_list) != 0:
            while True:
                sel_json = np.random.choice(candidate_list)
                bpm_beat = self.j2b_dict[sel_json]
                count += 1
                pos1 = np.where(np.array(bpm_beat["beat_positions"]) == 1)[0]
                pos1_len = pos1.shape[0]

                # At least one bar period
                if pos1_len > 2:
                    first_pos, last_pos = pos1[0], pos1[-1]
                    downbeats_duration = bpm_beat["downbeats"][-1] - bpm_beat["downbeats"][0]
                    break

                # Stop because not found
                if count >= 5:
                    break
        
        return sel_json, downbeats_duration
    
    def accom_beat_padding(self, waveform,
                           s_name,
                           dbs_duration):
        content = self.j2b_dict[s_name]
        start, end = content["downbeats"][0], content["downbeats"][-1]
        sel_waveform = waveform[int(start * self.sr) : int(end * self.sr)]

        if dbs_duration < self.thr:
            # print(f"start: {start}, end: {end}")
            cp_num = math.ceil(self.thr / dbs_duration)
            sel_waveform = np.concatenate([sel_waveform] * cp_num)
        
        waveform_thr = int(self.thr * self.sr) + 1
        return sel_waveform[:waveform_thr]

    def sv_beat_align(self, wav_sv, sel_json):
        downbeats = self.j2b_dict[sel_json + ".json"]["downbeats"]
        wav_seg = wav_sv[int(downbeats[0] * self.sr):int(downbeats[-1] * self.sr)]
        # print(f"downbeats[0:-1]: {downbeats}")
        rand_start = np.random.choice(downbeats[0:-1])
        rand_start_seg = wav_sv[int(rand_start * self.sr):int(downbeats[-1] * self.sr)]
        
        if rand_start_seg.shape[0] >= self.waveform_thr:
            output = rand_start_seg
        else:
            remain_len = self.waveform_thr - rand_start_seg.shape[0] - wav_seg.shape[0]
            if remain_len >= 0:
                padded_len = math.ceil(remain_len // wav_seg.shape[0]) + 1
            else:
                padded_len = 1
            # print(f"remain_len: {remain_len}, wav_seg.shape[0]: {wav_seg.shape[0]}, padded_len: {padded_len}")
            padded_wav_seg = np.concatenate([wav_seg] * padded_len)
            output = np.concatenate((rand_start_seg, padded_wav_seg))
        
        output = output[:self.waveform_thr]
        return output


        
if __name__ == "__main__":
    train_acc_path = "./dataset/split_dump_flac/train/non_vocals/"
    train_vocal_path = "./dataset/split_dump_flac/train/vocals/"
    b2j_path = "./dataset/split_dump_flac/train/bpm2json.json"
    j2b_path = "./dataset/split_dump_flac/train/json2bpm.json"

    with open(j2b_path, "r") as d:
        json2bpm_dict = json.load(d)
    
    BpmProCls = BpmProcessor(train_acc_path=train_acc_path,
                             json2bpm_path=j2b_path,
                             bpm2json_path=b2j_path,
                             sample_rate=16000,
                             threshold=64600)

    wav_path = "./dataset/split_dump_flac/train/non_vocals/0_0212_3.wav"
    json_name = os.path.basename(wav_path)[:-4] + ".json"
    wav, sample_rate = librosa.load(wav_path, sr=16000)

    bpm_n = json2bpm_dict[json_name]["bpm"]

    sel_json, dbs_duration = BpmProCls.sel_accom_from_bpm_group(bpm_n, "1")
    sel_wav = BpmProCls.load_audio_by_json(sel_json)
    padded_waveform = BpmProCls.accom_beat_padding(sel_wav, sel_json, dbs_duration)
    print(f"padded_waveform: {padded_waveform.shape}")

    bpm_res = json2bpm_dict["1_1344_11.json"]["bpm"]
    print(f"bpm_res: {bpm_res}")

    wav_path = os.path.join(train_vocal_path, "0_0321_4.flac")
    wav11, sample_rate = librosa.load(wav_path, sr=16000)
    
    wav_sv = BpmProCls.sv_beat_align(wav11, "0_0321_4")
    print(f"wav_sv: {wav_sv.shape}")
    