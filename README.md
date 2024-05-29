# Quantum Reinforcement Learning for Replenishment problem

We have developed a Quantum Reinforcement Learning (QRL) model to run on custom gym environment. We also have classical model (CRL) as well. QRL uses Qiskit for dealing with the quantum part and PyTorch to deal with ML side of things. CRL model uses the Keras package to implement RL.

## Description of the files/scripts

### Files necessary to run QRL and CRL

* rep_env.py - Script where the custom environment is defined.  
* helper.py - Script where helper functions are defined.
* data_edit - Folder where the processed datasets are present.

### Training/Testing files

* qiskit_QNNRL.py - Script where we have implemented QRL. This script will train and test a QRL model. Will also save the model and plots in the respective folders.
* qiskit_QNNRL_test.py Script to test pre-trained model. Run this script to quickly get results for QRL, it runs instantly, as it only simulates few hundred circuits.
* CNNRL.py - Script for CRL training and testing.  
* CNNRL_test.py - Script for testing a pre-trained model.
* aws_QRL.ipynb - Script we used to test QRL model on IonQ Aria.


### Extra files/folders

* models - Pre-trained models are present in this folder. 
* plots - Plots generated during training and testing are present in this folder.
* dataset.ipynb - Notebook for processing the datasets.
* Idea Summary-Replenishment of retail stores.pdf is a summarised document where anyone can get idea about problem statement, solution approach, results and future scalability.

> We have provided requirements.txt file containing all the required packages to run all the scripts and notebooks above.
