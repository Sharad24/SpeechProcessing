
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


# In[5]:


class AudioDataset(Dataset):
    def __init__(self, filename, directory, mfcc_features):
        self.filename = filename
        self.directory = directory
        self.features = mfcc_features
        self.file = pd.read_pickle(self.directory + self.filename)

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        audio_file = librosa.feature.mfcc(self.file.iloc[idx, 0], n_mfcc=self.features)
        if self.file.iloc[idx, 2] == 'female':
            gender = 1
        else:
            gender = 0
        sample = {'audio' : audio_file, 'target' : gender}
        return sample 


