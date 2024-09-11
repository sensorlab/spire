import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
import sys

filepath, savepath = sys.argv

with open("%s" % filepath, 'r') as f:
    num_lines = 1
    first_line = f.readline()
    j_line = json.loads(first_line)
    num_measurements = len(j_line["Measurements"])
    for _ in f:
        num_lines += 1

    f.seek(0)

    with h5py.File(savepath, "w") as outfile:
        dset = outfile.create_dataset("rss", (num_lines, num_measurements), dtype='d')
        timestamps = outfile.create_dataset("timestamp", (num_lines,), dtype='d')
        timeindex = outfile.create_dataset("timeindex", (num_lines,), dtype='d')
    
        for i, line in enumerate(f):
            j_line = json.loads(line)
            ts = datetime.datetime.strptime(j_line["Time"], '%Y-%m-%dT%H:%M:%S.%f').timestamp()
            t.append(ts)
        
            # use the h5py
            dset[i, :] = j_line["Measurements"]
            timestamps[i] = ts
            timeindex[i] = i

        print("Done!")