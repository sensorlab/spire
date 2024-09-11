import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
import sys

START_SAMPLE = 100
NUM_SAMPLES = 2048
R_F = 5_900_000_000
fftsize = 1024


#filepath = '25msps/'

filepath, savepath = sys.argv


def load_data_dict(filepath):
    '''Load data in dictionary for given filepath.'''
    files_list = sorted(glob.glob(filepath + '*.mat'))
    data_dict = {}
    
    for filename in files_list:
        name_start_idx = filename.index('/')
        name_end_idx = filename.index('_')
        tec_name = filename[name_start_idx+1:name_end_idx]

        if tec_name=='FiveG':
            tec_name = 'Five_G'
        if tec_name=='noise':
            tec_name = 'Noise'ве
        if tec_name=='wifi':
            tec_name='WiFi'
        print('Loading ' + tec_name + ' data.')
        
        with h5py.File(filename, 'r') as f:
            data = f[tec_name][:]
            data_dict.update({tec_name: data})
            
    return data_dict


def make_complex(raw_data):
    '''Make complex sample from given I/Q data.'''
    # Separate I and Q channels
    i_sample = raw_data[0:START_SAMPLE+NUM_SAMPLES:2]
    q_sample = raw_data[1:START_SAMPLE+NUM_SAMPLES+1:2]

    complex_sample = i_sample + 1j*q_sample
    
    return complex_sample


def get_subsamples(data_sample, fftsize=1024, stride=50):
    '''Derives subsamples from given sample for provided stride and fftsize.'''
    
    sample_len = data_sample.shape[0]
    subsample_end = fftsize
    subsample_start = 0
    derived_samples = []
    
    while subsample_end < sample_len:
        derived_samples.append(data_sample[subsample_start:subsample_end])
        subsample_start += stride
        subsample_end += stride
    
    derived_samples = np.array(derived_samples)
    
    return derived_samples

def get_subsamples_fft(data, fftsize):
    '''Calculates FFT of given batch of samples.'''
    subsamples_fft = []
    for i in range(data.shape[0]):
        complex_sample = make_complex(data[i, :])
        subsample_fft = 10*np.log10(np.absolute(np.fft.fftshift(np.fft.fft(complex_sample))))
        subsamples_fft.append(subsample_fft)
        
    subsamples_fft = np.array(subsamples_fft)
    return subsamples_fft


with h5py.File(savepath, 'a') as f:

    for label in list(data_dict.keys()):
        data = data_dict[label]
        print('Processing: ', label)

        for i in range(data.shape[1]):
        
            data_sample = data[:, i]
            data_subsamples = get_subsamples(data_sample, fftsize*2, stride=50)
            subsamples_fft = get_subsamples_fft(data_subsamples, fftsize)
        
            label = np.array(label, dtype=h5py.string_dtype())
            labels = np.repeat(label, subsamples_fft.shape[0])

            if not f.keys():
                f.create_dataset("fft", data=subsamples_fft, chunks=True, dtype=np.float32, maxshape=(None, None,))
                f.create_dataset("labels", data=labels, chunks=True, dtype=h5py.string_dtype(), maxshape=(None,))

            else:
                f['fft'].resize((f['fft'].shape[0] + subsamples_fft.shape[0]), axis=0)
                f['fft'][-subsamples_fft.shape[0]:] = subsamples_fft

                f['labels'].resize((f['labels'].shape[0] + labels.shape[0]), axis=0)
                f['labels'][-labels.shape[0]:] = labels
