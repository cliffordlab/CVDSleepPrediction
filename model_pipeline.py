import numpy as np
import pandas as pd
import os
import torch
from scipy.signal import resample, butter, filtfilt, iirnotch
from scipy import stats
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import pickle
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import h5py

from helper_functions import _load_model, notch_filter, zscore_signals
from PhysioNetChallenge2021_model import NN
from LSTM_model import *
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
print(DEVICE)

### Load model from PhysioNet Challeenge 2021 for arrhythmia classification (26 classes) and EMBEDDINGS extraction
model = _load_model("model/weights/",1)
model = model["classifier"].to(DEVICE)
num_classes = 26
model.eval()
print('Model for embeddings extraction loaded ...')

### LSTM output head for prediction of:
# 6 classes - Normal, AF, Stroke, MI, HF, Death
num_classes = 6
model_lstm = LSTM_architecture(num_classes=num_classes).to(DEVICE) 
state_dict = torch.load('model/weights/PSG_CVD.pth')
# Load the state dictionary into the model
model_lstm.load_state_dict(state_dict)
model_lstm.eval()
print('Model for CVD prediction loaded ...')


### Preprocessing
fs_target = 500
fs_original = 200 ### .h5 file preprocessed for CAISR v1 always sampled at 200 Hz
nyquist = (fs_target/2)
# Bandpass filtering 1 - 47Hz
b, a = butter(3, [0.05 / nyquist, 47 / nyquist], 'bandpass')

window_size = int(15 * fs_target) # 30s x sampling frequency
window_size_stage = int(15 * fs_original) # sampling frequency for sleep staging

# ECG lead indicator, we have 12 leads and in PSGs we usually have a modified lead II that is why leads = [0,1,0,0,0,0,0,0,0,0,0,0]
leads = np.zeros([1,12])
leads[0,1] = 1
leads = torch.tensor(leads).float().to(DEVICE)


def model_pipeline(recording, fs_original, stage = None):
    """
        Run the ECG model inference pipeline on a single recording.
    
        Parameters
        ----------
        recording : np.ndarray
            1D ECG signal of shape (N,), where N is the number of samples.
    
        fs_original : float or int
            Sampling frequency of the input ECG signal.
    
        stage : np.ndarray, optional
            Sleep stage annotations corresponding to the recording.
            Must have the same length as `recording`. If None, a vector
            of the same length as `recording` filled with value 9 
            (NOT SCORED) will be used.
            
            Encoding of sleep stages necessary: 
            N1: 3, N2: 2, N3: 1, REM: 4, WAKE: 5, NOT SCORED: 9
    
        
    
        Returns
        -------
        dict
            Dictionary containing predicted probabilities for:
            - AF (atrial fibrillation)
            - STROKE
            - MI (myocardial infarction)
            - HF (heart failure)
            - DEATH
    
        Notes
        -----
        The pipeline performs:
        1. Notch filtering to remove powerline noise.
        2. Resampling to the target sampling frequency.
        3. Bandpass filtering.
        4. Segmentation into 30-second windows.
        5. CNN inference to extract embeddings and per-window outputs.
        6. One-hot encoding of sleep stages.
        7. Sequence modeling with an LSTM to produce final outcome probabilities.
    """
    with torch.no_grad():
        if stage is None:
            stage = np.full(recording.shape[0], 9)
    
        recording = notch_filter(recording, fs_original, freq=60, Q=35)
    
        # Resample to 500 Hz
        num_samples = recording.shape[0]
        if fs_original != fs_target:
            recording = resample(recording, int(num_samples * fs_target / fs_original), axis=0)
        
        # Bandpass filter
        num_samples = recording.shape[0]
        recording = filtfilt(b, a, recording)
    
        # Number of 30 second windows
        num_windows = int(num_samples // window_size)
        
        # Prepare input shape needed for inference
        full_signal = np.zeros([12,recording.shape[0]])
        full_signal[1,:] = recording
            
        embeddings = []
        output_vector = []
        sleep_stages = []
        for w in range(num_windows):
            # Standardize 30 s window
            segment = zscore_signals(full_signal, w, window_size)
    
            # Select sleep stages from 30 s window
            sleep_stage = stage[w*window_size_stage:(w+1)*window_size_stage]
            # Extract human annotated sleep stage (mode)
            current_stage = stats.mode(sleep_stage).mode#[0]
            
            # Prepare input shape of the ECG needed for inference
            rec_out = np.zeros([12,8192])
            rec_out[:,0:7500] = segment
    
            # Create tensor
            xx = torch.tensor(rec_out).unsqueeze(0).unsqueeze(2).float().to(DEVICE) 
            
            # Inference
            y, p,embed = model(xx, leads) # output, sigmoid output, embeddings
            p = p.data.cpu().numpy()[0,:]
            embed = embed.data.cpu().numpy()[0,:]
            
            output_vector.append(p)
            embeddings.append(embed)
            sleep_stages.append(current_stage)
            
        ### Inference of the LSTM 
        output_vector = np.array(output_vector)
        embeddings = np.array(embeddings)
        sleep_stages = np.array(sleep_stages)
        
    
        # Check if we have sleep stages
        if sleep_stages.size == 0:
            # Return an empty array with correct shape (number of windows, 7)
            sleep_stages_out = np.zeros([embeddings.shape[0],7])
            sleep_stages_out[:,6] = 1 
        else:
            ### Creating One hot encoded sleep staging
            # Sleep staging N1 = 3, N2 = 2, N3 = 1, REM = 4, Wake  = 5, NOT SCORED = 0 or 9
            sleep_stages = sleep_stages.astype(int)
            # Change 9 to 6
            sleep_stages[sleep_stages == 9] = 6  # apply remapping
            # One hot encoded
            sleep_stages_out = np.eye(7)[sleep_stages]
    
        # Concatenate the inputs
        x = np.concatenate([embeddings, sleep_stages_out, output_vector], axis=1)
        # Create a tensor
        x = [torch.tensor(x)]
        # Prepare input
        lengths = torch.tensor([len(x) for x in x])
        ecg_padded = pad_sequence(x, batch_first=True)          # [B, N_max, 1, 7500]
        ecg_packed = pack_padded_sequence(ecg_padded, lengths, batch_first=True, enforce_sorted=False)
        lengths = lengths.float().to(DEVICE)
        ecg_packed = ecg_packed.float().to(DEVICE)
    
        # Inference of LSTM
        y,p, _ = model_lstm(ecg_packed, lengths)
    
    # Extract probabilities
    af_probability = p[1].item()
    stroke_probability = p[2].item()
    mi_probability = p[3].item()
    hf_probability = p[4].item()
    death_probability = p[5].item()
    return {'AF': af_probability, 
           'STROKE': stroke_probability,
           'MI': mi_probability, 
           'HF': hf_probability, 
           'DEATH': death_probability}