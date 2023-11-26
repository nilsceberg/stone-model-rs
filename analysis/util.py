import numpy as np

def reconstruct_path(flight):
    physical_states = np.array(flight["physical_states"])
    path = [[0,0]] + np.cumsum(physical_states[:,0:2], axis=0)
    return path


nature_single = 50 #89.0 / 25.4
figsize = (nature_single, nature_single)

n_mem = 16
