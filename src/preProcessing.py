'''
Author: Github: banishdeep07
		Mail:	bojha@stevens.edu
'''

#Libraries import
import os
import glob
import pandas as pd
import numpy as np
from VARIABLES import *
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def processTrainFiles(fileDirectory):
	file = pd.read_csv(fileDirectory)
	columns = file.columns

	#assign label correspondingly,unless data is shuffled, it holds 
	if LABEL_AND_TRAIN_ON_SAME_TABLE:
		if not LABEL_IS_LAST_COLUMN_IN_TRAIN:
			label = file[LABEL_COLUMN]
			file.drop(LABEL_COLUMN, axis = 'columns', inplace = True)
		else:
			label = file.iloc[:,-1]
			file.drop(file.columns[-1], axis = 'columns', inplace = True)

	file.drop(columns[0], axis = 'columns', inplace = True) #assuming 1st column is ID which is redundant for pattern recognition

	global trainData_Y
	trainData_Y=trainData_Y.append(label, ignore_index=True)
	
	global trainData_X
	trainData_X = trainData_X.append(file, ignore_index=True)


def processTrainTestFiles(ROOT_DIRECTORY):
	global trainData_Y
	global trainData_X
	global testData_X

	testData_X=pd.read_csv(TEST_DIRECTORY + "/" + os.listdir(TEST_DIRECTORY)[0]) # Assuming single test file

	columns = trainData_X.columns
	print("########################Before file Processing#############################")
	trainData_X.info()

	
	for eachColumn in columns:
		#Check NA all here
		if sum((trainData_X[eachColumn].isna())) > 0:
			if sum((trainData_X[eachColumn].isna())) > COLUMN_NA_THRESHOLD * trainData_X.shape[0]:
				trainData_X.drop(eachColumn, axis = 'columns', inplace = True)
			else:
				trainData_X[eachColumn+'_NA'] = trainData_X[eachColumn].isna() # creating new column if NA, this might give importnant information
				testData_X[eachColumn+'_NA'] = testData_X[eachColumn].isna()
				try:
					trainData_X[eachColumn].fillna(trainData_X[eachColumn].mean(), inplace = True) # if datatype is permissible with mean calculation
					testData_X[eachColumn].fillna(testData_X[eachColumn].mean(), inplace = True)
				except:
					trainData_X[eachColumn].fillna(lambda x: random.choice(trainData_X[trainData_X[column] != np.nan][eachColumn]), inplace=True) # else select random not null values
					testData_X[eachColumn].fillna(lambda x: random.choice(testData_X[testData_X[column] != np.nan][eachColumn]), inplace=True) # else select random not null values
					
	#save correlation matrix
	sns.set(rc={'figure.figsize':(50,50)})
	corr = trainData_X.corr()
	corr.to_csv(OUTPUT_DIRECTORY + '01.correlationAmongVariables.csv')
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
	plt.savefig(OUTPUT_DIRECTORY+'01.correlationAmongVariables.png', bbox_inches='tight', pad_inches=0.0)

	columns = trainData_X.columns
	categoricalColumns=[]
	for eachColumn in columns:
		#change to categorical if distinct value is less than threshold
		if (trainData_X[eachColumn].value_counts().shape[0] / trainData_X.shape[0]) < DISCRETE_THRESHOLD:
			trainData_X[eachColumn] = trainData_X[eachColumn].astype('category')
			categoricalColumns.append(eachColumn)

	for eachCategory in categoricalColumns:
	    LabelEncoderInstance = LabelEncoder()
	    try: # if test is defined
	    	dataToEncode = (pd.concat([trainData_X.loc[:,eachCategory],testData_X.loc[:,eachCategory]],axis = 0)).apply(str)
	    	LabelEncoderInstance.fit(dataToEncode)
	    	trainData_X.loc[:,eachCategory] = LabelEncoderInstance.transform((trainData_X.loc[:,eachCategory]).apply(str))   
	    	testData_X.loc[:,eachCategory] = LabelEncoderInstance.transform((testData_X.loc[:,eachCategory]).apply(str))
	    except:
	    	LabelEncoderInstance.fit(trainData_X.loc[:,eachCategory])
	    	trainData_X.loc[:,eachCategory] = LabelEncoderInstance.transform((trainData_X.loc[:,eachCategory]).apply(str))

	print("########################After file Processing#############################")
	trainData_X.info()
	#print(trainData_X.shape)
	#print(testData_X.shape)

def process():
	global trainData_X
	global trainData_Y
	global testData_X

	trainData_X = pd.DataFrame()
	trainData_Y = pd.DataFrame()
	testData_X = pd.DataFrame()

	if  not MULTIPLE_FILE_TRAIN:
		trainFiles = [TRAIN_DIRECTORY + os.listdir(TRAIN_DIRECTORY)[0]]
	else:
		trainFiles = [TRAIN_DIRECTORY + x for x in glob.glob(TRAIN_DIRECTORY+'*.csv')]


	#for each File in trainFiles, identified by blob
	for eachFile in trainFiles:
		processTrainFiles(eachFile)
	trainData_Y = trainData_Y.transpose()

	processTrainTestFiles(ROOT_DIRECTORY)

	trainData_X.to_pickle(OUTPUT_DIRECTORY + '00.trainData_X.pickle')
	testData_X.to_pickle(OUTPUT_DIRECTORY + '00.testData_X.pickle')
	trainData_Y.to_pickle(OUTPUT_DIRECTORY + '00.trainData_Y.pickle')