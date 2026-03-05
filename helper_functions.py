# Helper functions to run Cardiovascular Disease Prediction algorithm 

################################################################################

from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pickle
from pathlib import Path
from PhysioNetChallenge2021_model import NN
import torch
import warnings
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Generic function for loading a model.
def _load_model(model_directory,id):
    filename = Path(model_directory,f'MODEL_{id}.pickle')
    model = {}
    with open(filename, 'rb') as handle:
        input = pickle.load(handle)

    model['classifier'] = NN(nOUT=26).to(DEVICE)
    model['classifier'].load_state_dict(input['state_dict'])
    #model['classifier'].eval()
    model['thresholds'] = input['thresholds']
    model['classes'] = input['classes']
    return model


def notch_filter(data, fs, freq, Q):
    """
    Apply a notch filter to the given data.

    Parameters:
    data: numpy array, the input signal
    fs: float, the sampling frequency of the signal
    freq: float, the frequency to remove (e.g., 50 Hz or 60 Hz)
    Q: float, quality factor - determines the bandwidth around the target frequency
    
    Returns:
    filtered_signal: numpy array, the filtered signal
    """
    # Design the notch filter
    b, a = iirnotch(freq, Q, fs)
    
    # Apply the notch filter to the data
    filtered_signal = filtfilt(b, a, data)
    
    return filtered_signal

def zscore_signals(input,i, window_size):
    output = input[:, i * window_size : (i + 1) * window_size]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(output, axis=-1, keepdims = True)
        std = np.nanstd(output, axis=-1, keepdims = True)
        output = (output - mu) / std
        output = np.nan_to_num(output)
    return output