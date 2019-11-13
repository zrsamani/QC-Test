# QC-Test
# Zahra Riahi Samani
# Center for Biomedical Image Computing and Analytics: CBICA
# University of Pennsylvania
# November 2019

This directory includes:
QCTest.py It loads the pre-trained models and run in on the sample of test data
QCTrain.py It can be used to traine on Own model. The data set should contain 2-slices of artifact-free and artifactual slices which are converted to jpg format.
models it contained the pre-trained models
Results:The place where the result is written
Test: contains the test data


Requirenment:
you need to install these packages:
keras, tensorflow, dipy, cv2, PIL, Sklearn

How to run:
QCTest --axial will run the model on axial data, prints the accuracy,precision and recall and saves the output lables at Results folder.
QCTest --saggital will run the model on sagittal data, prints the accuracy,precision and recall and saves the output lables at Results folder.
output labels contains three columns: imagename, true lable, predicted label.

QCTrain --axial    will train the axial models based on the trining data  which should be stored in data folder
QCTran  --sagittal will train the sagittal models based on the trining data  which should be stored in data folder


The expected output for axial model:
Found 98 images belonging to 2 classes.
Computing features
features computed
Running Axial Model  
Accurcay: 0.989795918367
Recall: 0.979591836735
Precision: 1.0

The expected output for saggital model:

Found 98 images belonging to 2 classes.
Computing features
features computed
Running Sagittal Model
Accurcay: 0.959183673469
Recall: 0.959183673469
Precision: 0.959183673469
