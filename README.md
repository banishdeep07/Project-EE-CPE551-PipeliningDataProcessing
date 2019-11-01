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
