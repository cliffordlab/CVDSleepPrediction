# Prediction of cardiovascular outcomes from overnight polysomnography

## What's in this repository?

This repository contains a simple example that illustrates how to format an input data for deep learning model predicting incident atrial fibrillation (AF), stroke, myocardial infarction (MI), heart failure (HF) and death. 

## How do I run these scripts?

First, you can download [example data](https://drive.google.com/drive/folders/1D8F9Z1QtKTUTeZbL1XDF9w1CjJOv4gGj?usp=share_link) and put them in data folder in the main directory of this github repo called 'data_example'. 

Second, you can install the dependencies for these scripts by running

    pip install -r requirements.txt
    
Next, you can run 

    run_inference.ipynb


which loads example_data and calls function **model_pipepline** from **model_pipeline.py** which

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

## Citation
The paper describing the code and data is included in the Documents folder and is currently under review in SLEEP. 

## The data used for algorithm development and testing are from The Human Sleep Project
 Sun, H., Ganglberger, W., Nasiri, S., Gupta, A., Ghanta, M., Moura Junior, V., Cash, S., Stone, K., Zhang, Z., Ganjoo, G., Nassi, T. E., Wei, R., Meulenbrugge, E., Au, R., Clifford, G., Trotti, L. M., Hwang, D., Mignot, E., Katwa, U., & Westover, M. B. (2023). The Human Sleep Project (version 2.0). Brain Data Science Platform. https://doi.org/10.60508/qjbv-hg78.
