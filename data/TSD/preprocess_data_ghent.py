import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
import sys
import glob

dirpath = sys.argv[1]
savepath = sys.argv[2]

NUM_SAMPLES = 400_000
F_S = 1_000_000
R_F = 2_412_000_000
fftsize = 1024

def get_labeled_data(filename):
    # Load original data from file
    # ============================
    start_idx = filename.index('ghent/')
    end_idx = filename.index('_g')
    
    with open(filename, 'rb') as file:
        fileContent = file.read()

    raw_data = data

    # Separate I and Q channels
    i_sample = raw_data[0:len(raw_data):2]
    q_sample = raw_data[1:len(raw_data)+1:2]

    complex_sample = i_sample + 1j*q_sample

    nfft = int(np.floor(len(complex_sample)/fftsize))
    spacematrix = np.zeros((nfft, fftsize))
    labels = []

    for i in range(nfft):
        temp_data = complex_sample[i*fftsize:(i+1)*fftsize]
        spacematrix[i, :] = 10*np.log10(np.absolute(np.fft.fftshift(np.fft.fft(temp_data))))
        labels.append(filename[start_idx+6:end_idx])

    labels = np.array(labels, dtype=h5py.string_dtype())
    return spacematrix, labels


filenames = sorted(glob.glob(f"{dirpath}*.bin")) 

for filename in filenames:
    with open(filename, 'rb') as file:
        fileContent = file.read()

    # Transform binary to real
    data = np.frombuffer(fileContent, dtype=np.float32)

    spacematrix, labels = get_labeled_data(filename)


    with h5py.File(savepath, 'a') as f:
        if not f.keys():
            f.create_dataset("rss", data=spacematrix, chunks=True, dtype=np.float32, maxshape=(None, None,))
            f.create_dataset("labels", data=labels, chunks=True, dtype=h5py.string_dtype(), maxshape=(None,))
        
        else:
            f['rss'].resize((f['rss'].shape[0] + spacematrix.shape[0]), axis=0)
            f['rss'][-spacematrix.shape[0]:] = spacematrix

            f['labels'].resize((f['labels'].shape[0] + labels.shape[0]), axis=0)
            f['labels'][-labels.shape[0]:] = labels
    
print('Done')
    

