# ProjectEE551-PipeliningGeneralDSProcessing
This project is for an educational practice. This is the project as the requirement for the completion of course CPE/EE 551, Engineer Progmng/Applc in Python

## Introduction
This proposal outlines the theme of the project as requirement for the course completion for EE/CPE 551 Engineering Application and Programming in Python.
Instructor for the course: Prof. Sergul Aydore

## Motivation
In Data Science Application, there are lot of intersection portion for initial data processing and cleaning during different cases. Specially for open source competitions like kaggle, the data processing, wrangling etc involve similar flow of the controls as the data are usually structured or semi-structured. It is tedious for anyone to go through similar steps again and again as they participate in multiple competitions. So, this project tries to create a pipeline for intersection steps among various model construction and deployment.

## Description
For time constraints consideration, this project will limit the scope to organized flat files and one submission file required. The end product of the project is proposed to do the following tasks as pipeline:
* Dropping the columns where the NA values are greater than some specified threshold
* For each column, making another column to preserve NA data [because absence of data may be important information for models]
* Efficient data type usage to increase efficiency [for instance, object datatype instead of categorical, etc]
* For categorical columns, one hot encoding [for identification of categorical column, some brute percentage is used as threshold]
* Random split of test-train data [weighted/ stratified if unbalanced]
* Implementation of some models [random search if no parameters specified]
* Ensemble of models and,
* Output if submission required

The open stub file will be created with parameters specification if user wants to override default parameters like location of files, the target column name, hyperparameters or searchspace for models in dictionary form, ensemble methods, etc.

## Testing
The testing of the pipeline will be performed using two of the kaggle competions where training and testing files are flat files. One competetion dataset will be used as baseline for pipelining and other will be used for validation during development phase.

## December 01, on submission comments

## Changes on proposal
* Label encoding instead of one hot encoding
* No random search to make efficient running [for this submission]
* Ensemble not performed, but individual output for the each model is saved

## Libraries required
* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* lightgbm
* xgboost

## Processes to run this repos (with assumption all required packages are installed)
1. Cloning repos
2. Putting the required files in required directories
* train file in directory: data/train/
* test file in directory: data/test/ [if needed to, optional]
* sample_submission file in directory: data/sample_submission/ [if needed to, optional]
3. Opening src/VARIABLES.py in any preferred editor and editting the variables as per required. This file acts as control over program and model training
4. running src/main.py and waiting till completion [time consuming process]
5. Checking outputs on directory: outputs/
6. If sample_submission provided and required parameters specified in step 3, browsing to outputs/Predictions/ and making submissions (on kaggle) if desired
7. Based on result and outputs, customized models can be used to train and produce output

## For an instance, I have included outputs as:
1. The very small data subset is taken from original source: https://www.kaggle.com/c/santander-customer-transaction-prediction. And default classifiers are used
The changes mde from original dataset includes:
* Only selecting first 1000 rows of training dataset
* Inserting random empty elements in two columns
* For testing on 25 instances only for test data, only 25 rows were included. Sample Submission file is adjusted accordingly

## This reduction of data is made only to portray usability in efficient manner. This might induce __warnings__ and __overfitting__. To see the original outputs except pickle output, please check this shared drive link:
[https://drive.google.com/open?id=1iRhp4WPv2WedqNJfCPykGKMwVhFwaoa1]

2. The run of models as proposed create the following outputs:
* 00.trainData_X.pickle
* 00.trainData_Y.pickle
* 00.testData_X.pickle
#### These generation of 3 pickle files are to avoid repetition of processing units. If the program run finds pickle files, it directly jumps to training phase
* 01.correlationAmongVariables.jpg
* 01.correlationAmongVariables.csv
#### Above two files help in developing quick insight on data, importance of NA data
* LGBMClassifierAUC.png
* MLPClassifierAUC.png
* XGBClassifierAUC.png
#### These three output plots for AUC help in analysis of performance per cross validation folds. SImilarly, Variance Explained and R2 measure is used for Regression problems
* Predictions/LGBMClassifieroutputPrediction.csv
* Predictions/MLPClassifieroutputPrediction.csv
* Predictions/XGBClassifieroutputPrediction.csv
#### These three output files can be used to make submission [on kaggle] if desired to evaluate

## Also this workflow is verified using data without joins from source: [https://www.kaggle.com/c/ieee-fraud-detection/data] and the run is successful without any changes to be made except some controls in src/VARIABLES.py


## Future Enhancements
* Development can be extended to incorporate multiple types of training data like multiple input files requiring join [though, those not requiring join are incorporated on submission on December 01]
* Inclusion of other unstructured-data-types like images, text, etc training as well on pipeline
* More efficient processing and pipelining
* Random Search [though can be triggered through custom model on submission on December 01]
