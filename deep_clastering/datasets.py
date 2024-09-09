import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

GHENT_DATASET_RSS_MIN = -55.7776
GHENT_DATASET_RSS_MAX = 9.9656

LOGATEC_DATASET_RSS_MIN = -133.0
LOGATEC_DATASET_RSS_MAX = -68.0

ITS_DATASET_RSS_MIN = 20.697266
ITS_DATASET_RSS_MAX = 60.435753


class LabeledGhentDataset(Dataset):
    
    name = 'LabeledGhentDataset'
    
    def __init__(self, dataset_path, transform=None, scale=False, limit=None, labels=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.scale = scale
        self.limit = limit
        self.RSS_MIN = GHENT_DATASET_RSS_MIN
        self.RSS_MAX = GHENT_DATASET_RSS_MAX
        self.labels_dict = {'dvbt':0, 'lte':1, 'wf':2}
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            #Obtain max time range and max bandwidth range
            self.data_length = fp['rss'].shape[0]
            self.data_width = fp['rss'].shape[1]
                
        if not labels is None:
            self.labels = labels
        else:
            self.labels = np.zeros(self.data_length, dtype=int)
    
    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data_length
            
    def __getitem__(self, idx):
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            sample = fp['rss'][idx]
            sample_label = self.labels_dict[bytes.decode(fp['labels'][idx])]
            
        if self.scale:
            sample = (sample-self.RSS_MIN)/(self.RSS_MAX-self.RSS_MIN)
            
        if self.transform:
            sample = self.transform(sample, dtype=torch.float)
            sample = sample.unsqueeze(0)
            
        return sample, sample_label

class LogatecDataset(Dataset):
    
    name = 'LogatecDataset'
    
    def __init__(self, dataset_path, transform=None, scale=False, limit=None, labels=None):
        self.dataset_path = dataset_path
        self.transform = transform
        # Constants predetermined
        self.RSS_MIN = LOGATEC_DATASET_RSS_MIN
        self.RSS_MAX = LOGATEC_DATASET_RSS_MAX
        self.datasetpath = dataset_path
        self.scale = scale
        self.limit = limit
        
        with h5py.File(self.datasetpath, mode="r", swmr=True) as fp:
            #Obtain max time range and max bandwidth range
            self.data_length = len(fp["timestamp"])
            self.data_width = len(fp['rss'][0])
                
        if not labels is None:
            self.labels = labels
        else:
            self.labels = np.zeros(self.data_length, dtype=int)
    
    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data_length
            
    def __getitem__(self, idx):
        
        with h5py.File(self.datasetpath, mode="r", swmr=True) as fp:
            sample = fp['rss'][idx]
            
        if self.scale:
            sample = (sample-self.RSS_MIN)/(self.RSS_MAX-self.RSS_MIN)
            
        if self.transform:
            sample = self.transform(sample, dtype=torch.float)
            sample = sample.unsqueeze(0)
            
        return sample, self.labels[idx]
    

class LabeledITSDataset(Dataset):
    
    name = 'LabeledITSDataset'
    
    def __init__(self, dataset_path, transform=None, scale=False, limit=None, labels=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.scale = scale
        self.limit = limit
        self.RSS_MIN = ITS_DATASET_RSS_MIN
        self.RSS_MAX = ITS_DATASET_RSS_MAX
        self.labels_dict = {'CV2X': 0, 'Five_G': 1, 'ITSG5': 2, 'LTE': 3, 'WiFi': 4, 'Noise': 5}
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            #Obtain max time range and max bandwidth range
            self.data_length = fp['fft'].shape[0]
            self.data_width = fp['fft'].shape[1]
            self.gt_labels = fp['labels']
                
        if labels:
            self.labels = labels
        else:
            self.labels = np.zeros(self.data_length, dtype=int)
    
    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data_length
            
    def __getitem__(self, idx):
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            sample = fp['fft'][idx]
            sample_label = self.labels_dict[bytes.decode(fp['labels'][idx])]
            
        if self.scale:
            sample = (sample-self.RSS_MIN)/(self.RSS_MAX-self.RSS_MIN)
            
        if self.transform:
            sample = self.transform(sample, dtype=torch.float)
            sample = sample.unsqueeze(0)
            
        return sample, sample_label
