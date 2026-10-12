# Prediction of cardiovascular outcomes from overnight polysomnography

## What's in this repository?

This repository contains a simple example that illustrates how to format an input data for deep learning model predicting incident atrial fibrillation (AF), stroke, myocardial infarction (MI), heart failure (HF) and death. 

## How do I run these scripts?

First, you can download [example data](https://drive.google.com/drive/folders/1D8F9Z1QtKTUTeZbL1XDF9w1CjJOv4gGj?usp=share_link) and place it in the data folder located in the main directory of this GitHub repository called 'data_example'. 

Second, you can install the dependencies for these scripts by running

    pip install -r requirements.txt
    
Next, you can run 

    run_inference.ipynb


which loads example_data and calls function [model_pipeline](https://github.com/zkoscov/CVDSleepPrediction/blob/9011c17e6e9ba9a6ab09bd9942f5dc4d95e267de/model_pipeline.py#L56) from **model_pipeline.py** which

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

# Citation
## When using this resource please cite the paper describing the code and data:

Zuzana Koscova, Samaneh Nasiri, Matthew A Reyna, Neal K Bhatia, Ali Bahrami Rad, Reza Sameni, Haoqi Sun, Niels Turley, Umakanth Katwa, Dennis Hwang, Katie L Stone, Lynn M Trotti, Robert J Thomas, Emmanuel Mignot, Brandon M Westover, Gari D Clifford, Prediction of cardiovascular outcomes using electrocardiography from overnight polysomnography, Sleep, 2026;, zsag229, https://doi.org/10.1093/sleep/zsag229

## The data used for algorithm development and testing are from The Human Sleep Project:

Qichen Li, Shenghan Wen, Haoqi Sun, Wolfgang Ganglberger, Ayush Tripathi, Niels Turley, Samuel Waters, Arnav Gupta, Aditya Gupta, Manohar Ghanta, Bruce Nearing, Han Wu, Katie Stone, Chad Robichaux, Zhiyong Zhang, Qiao Li, Gauri Ganjoo, Christine Tsien Silvers, Bharath Gunapati, Kiran Maski, Samaneh Nasiri, Dennis Hwang, Lynn Marie Trotti, Umakanth Katwa, Gari Clifford, Emmanuel Mignot, Robert J Thomas, Michael Brandon Westover, The human sleep project: a multi-center clinical polysomnography dataset across the human lifespan, Sleep, 2026;, zsag215, https://doi.org/10.1093/sleep/zsag215
