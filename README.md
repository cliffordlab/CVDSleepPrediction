# Prediction of cardiovascular outcomes from overnight polysomnography

## What's in this repository?

This repository contains a simple example that illustrates how to format an input data for deep learning model predicting incident atrial fibrillation (AF), stroke, myocardial infarction (MI), heart failure (HF) and death. 

## How do I run these scripts?

First, you can download [example data](https://github.com/physionetchallenges/python-example-2026?tab=readme-ov-file#how-do-i-create-data-for-these-scripts) and put them in data folder in the main directory of this github repo called 'data_example'. 

Second, you can install the dependencies for these scripts by running

    pip install -r requirements.txt


which calls function **model_pipepline** from **model_pipeline.py** which

 Run the ECG model inference pipeline on a single recording.
    
        Input
        ----------
        recording : np.ndarray
            1D ECG signal of shape (N,), where N is the number of samples. The minimum time of ECG should be at least 1 hour for code to work properly.
    
        fs_original : float or int
            Sampling frequency of the input ECG signal.
    
        stage : np.ndarray, optional
            Sleep stage annotations corresponding to the recording.
            Must have the same length as `recording`. 
            If None, a vector of the same length as `recording` filled with value 9 
            (NOT SCORED) will be used.
            
            **Encoding** of sleep stages necessary:
            N1: 3, N2: 2, N3: 1, REM: 4, WAKE: 5, NOT SCORED: 9
    
        
    
        Returns
        -------
        dict
            Dictionary containing predicted probabilities for future:
            - AF (atrial fibrillation)
            - STROKE
            - MI (myocardial infarction)
            - HF (heart failure)
            - DEATH

