import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm_notebook 

from utils import World, Converter
from hparams import hparams

def mcep_statistics(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 0)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 0, keepdims = False)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 0, keepdims = False)
    return coded_sps_mean, coded_sps_std

def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    return log_f0s_mean, log_f0s_std

class VoiceDataset(Dataset):
    
    def __init__(self, root_dir, source_limit, transform=None):
        
        self.world = World()
        
        self.speakers = []
        for d_name in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d_name)):
                self.speakers.append(d_name)
        print(self.speakers)
        
        self.data = []
        self.label = []
        for i, d_name in enumerate(self.speakers):
            data_dir = os.path.join(root_dir, d_name)
            if os.path.isfile(os.path.join(data_dir, d_name + "_mcep.pickle")):
                with open(os.path.join(data_dir, d_name + "_mcep.pickle"), mode="rb") as data:
                    mceps = pickle.load(data)
                    mceps = mceps[:source_limit]
                    self.data.extend(mceps)
                    self.label.extend(np.ones((len(mceps)))*i)
                print("[{}] mcep loaded.".format(d_name))
            else:
                mceps = []
                f0s = []
                for f in tqdm_notebook(os.listdir(data_dir)):
                    if not ".wav" in f:
                        continue
                    file_path = os.path.join(data_dir, f)
                    wav, _ = librosa.load(file_path, sr=hparams.fs)
                    if len(wav) <= 10:
                        continue
                    wav, _ = librosa.effects.trim(wav)
                    wav = wav.astype(np.double)
                    f0, spec, ap = self.world.analyze(wav)
                    mcep = self.world.mcep_from_spec(spec)
                    mcep = mcep.reshape(mcep.shape[0], mcep.shape[1], 1)
                    if mcep.shape[0] < 128:
                        continue
                    mceps.append(mcep)
                    f0s.append(f0)
                # mceps = mceps[:source_limit]
                self.data.extend(mceps)
                self.label.extend(np.ones((len(mceps)))*i)
                with open(os.path.join(data_dir, d_name + "_mcep.pickle"), mode='wb') as f:
                    pickle.dump(mceps, f)
                log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
                mceps_mean, mceps_std = mcep_statistics(mceps)
                np.savez(os.path.join(data_dir, d_name + "_norm.npz"), log_f0s_mean=log_f0s_mean, 
                         log_f0s_std=log_f0s_std, mceps_mean=mceps_mean, mceps_std=mceps_std)
                print("[{}] voices converted.".format(d_name))
                    
        self.transform = transform
        self.converter = Converter(root_dir, self.speakers)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        data = self.data[idx]
        label = self.label[idx]
        label = int(label)
        
        data = self.converter.forward_process(data, label)

        if self.transform:
            data, label = self.transform((data, label))
        return data, label
    
    
### Transformer ###
    
class RandomCrop(object):

    def __init__(self):
        output_size = hparams.crop_size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, inputs):
        data, label = inputs
        
        h, w = data.shape[:2]
        # new_h, new_w = self.output_size
        new_w, new_h = self.output_size
        
        if h - new_h != 0:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if w - new_w != 0:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        data = data[top: top + new_h, left: left + new_w]
        
        return (data, label)
    
class ToTensor(object):

    def __call__(self, inputs):
        data, label = inputs

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 1, 0))
        # data = data.transpose((2, 0, 1))
        
        return (torch.tensor(data,dtype=torch.float), torch.tensor(label, dtype=torch.long))
