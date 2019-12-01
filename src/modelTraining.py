'''
Author: Github: banishdeep07
		Mail:	bojha@stevens.edu
'''
from VARIABLES import *

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import pandas as pd
import matplotlib.pyplot as plt


def compute_roc_auc(model,index): #for each cross validation evaluation and to plot later
    y_predict = model.predict_proba(trainData_X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(trainData_Y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def evaluateRegressor(model, index):
	y_predict = model.predict(trainData_X.iloc[index])
	return explained_variance_score(trainData_Y.iloc[index], y_predict), r2_score(trainData_Y.iloc[index], y_predict)

def TrainClassifierModelCV(trainData_X, trainData_Y, model):
	falsePositiveRates, truePositiveRates, areaUnderCurves = [], [], []
	cv = StratifiedKFold(n_splits=CROSS_VALIDATION_NUMBER, random_state=1884664, shuffle=True) #Shuffle done here, so careful attainment after this
	#print(type(trainData_X), trainData_Y.shape)
	loopTrack=0
	print("Training with model: "+type(model).__name__)
	for (train, test), i in zip(cv.split(trainData_X, trainData_Y), range(CROSS_VALIDATION_NUMBER)):
	    print('{} of CV folds {} running.........'.format(loopTrack+1, CROSS_VALIDATION_NUMBER))
	    if type(model).__name__ == 'XGBClassifier' or type(model).__name__ == 'LGBMClassifier':
	    	model.fit(trainData_X.iloc[train,:], trainData_Y.iloc[train,:],eval_metric='auc')
	    else:
	    	model.fit(trainData_X.iloc[train,:], trainData_Y.iloc[train,:])
	    _, _, trainAUC = compute_roc_auc(model,train)
	    fpr, tpr, testAUC = compute_roc_auc(model,test)
	    print('For this fold: trainAUC:{} and testAUC:{}'.format(trainAUC, testAUC))
	    areaUnderCurves.append((trainAUC, testAUC))
	    falsePositiveRates.append(fpr)
	    truePositiveRates.append(tpr)
	    loopTrack+=1
	return falsePositiveRates, truePositiveRates, areaUnderCurves, model

def TrainRegressorModelCV(trainData_X, trainData_Y, model):
	varianceExplained, RSquaredMeasure = [], []
	cv = StratifiedKFold(n_splits=CROSS_VALIDATION_NUMBER, random_state=1826423, shuffle=True) #Shuffle done here, so careful attainment after this
	#print(type(trainData_X), trainData_Y.shape)
	loopTrack=0
	print("Training with model: "+type(model).__name__)
	for (train, test), i in zip(cv.split(trainData_X, trainData_Y), range(CROSS_VALIDATION_NUMBER)):
	    print('{} of CV folds {} running.........'.format(loopTrack+1, CROSS_VALIDATION_NUMBER))
	    model.fit(trainData_X.iloc[train,:], trainData_Y.iloc[train,:])
	    trainEV, trainR2 = evaluateRegressor(model,train)
	    testEV, testR2 = evaluateRegressor(model,test)
	    print('For this fold: trainEV:{} and testEV:{}\n trainR2:{} and testR2:{}'.format(trainEV, testEV, trainR2, testR2))
	    varianceExplained.append(testEV)
	    RSquaredMeasure.append(testR2)
	    loopTrack+=1
	return varianceExplained, RSquaredMeasure, model

def saveROCPlots(fprs, tprs, aucs,model_name):
	plt.figure()
	lw = 1
	colors = ['black','red','blue','darkorange','green']
	for i in range(0,CROSS_VALIDATION_NUMBER):
	    plt.plot(fprs[i], tprs[i], color=colors[i%5],
	             lw=lw, label='ROC curve (area = %0.5f)' % aucs[i][1])
	    
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.savefig(OUTPUT_DIRECTORY+ model_name + 'AUC.png', bbox_inches='tight', pad_inches=0.0)

def saveVERSMPlots(varianceExplained, RSquaredMeasure, model_name):
	plt.figure()
	plt.plot(RSquaredMeasure)
	plt.xlabel('CV fold no')
	plt.ylabel('R Squared Measure')
	plt.savefig(OUTPUT_DIRECTORY+ model_name + 'RSquaredMeasure.png', bbox_inches='tight', pad_inches=0.0)


	plt.figure()
	plt.plot(varianceExplained)
	plt.xlabel('CV fold no')
	plt.ylabel('Variance Explained')
	plt.savefig(OUTPUT_DIRECTORY+ model_name + 'varianceExplained.png', bbox_inches='tight', pad_inches=0.0)

def trainModelsAndCreateSubmission():
	if SAMPLE_SUBMISSION_PROVIDED:
		sampleSubmission = pd.read_csv(SAMPLE_SUBMISSION_DIRECTORY + 'sample_submission.csv')
		column=sampleSubmission.columns
		
		testData_X = pd.read_pickle(OUTPUT_DIRECTORY + '00.testData_X.pickle')

		indexName=testData_X.iloc[:,0]
		testData_X.drop(column[0], axis = 'columns', inplace = True) #ready to predict now


	global trainData_X
	global trainData_Y

	trainData_X = pd.read_pickle(OUTPUT_DIRECTORY + '00.trainData_X.pickle')
	trainData_Y = pd.read_pickle(OUTPUT_DIRECTORY + '00.trainData_Y.pickle')

	if PROBLEM_TYPE == "Classification":
		if CUSTOMMODELS==False:
			MODELS={MLPClassifier:{},XGBClassifier:{}, LGBMClassifier:{}}
		for model,params in MODELS.items():
		    clf=model()
		    clf.set_params(**params)
		    falsePositiveRates, truePositiveRates, areaUnderCurves, trainedModel=TrainClassifierModelCV(trainData_X, trainData_Y, clf)
		    saveROCPlots(falsePositiveRates, truePositiveRates, areaUnderCurves,type(clf).__name__)
		    output = (trainedModel.predict_proba(testData_X))[:,1] #with assumption evaluation metrics used is AUC, works for binary classification
		    #output = (trainedModel.predict(testData_X))[:,1] #with assumption evaluation metrics used is Entropy for classification
		    toSubmitCSV = pd.DataFrame({column[0]:indexName,column[1]:output})
		    toSubmitCSV.to_csv(OUTPUT_DIRECTORY+'Predictions/'+type(clf).__name__+'outputPrediction.csv', index=None)


	elif PROBLEM_TYPE == "Regression":
		if CUSTOMMODELS==False:
			MODELS={MLPRegressor:{},XGBRegressor:{}, LGBMRegressor:{}}
		for model,params in MODELS.items():
		    clf=model()
		    clf.set_params(**params)
		    varianceExplained, RSquaredMeasure, trainedModel=TrainRegressorModelCV(trainData_X, trainData_Y, clf)
		    saveVERSMPlots(varianceExplained, RSquaredMeasure,type(clf).__name__)
		    output = (trainedModel.predict(testData_X)) #with assumptional evaluation metrics used is AUC
		    toSubmitCSV = pd.DataFrame({column[0]:indexName,column[1]:output})
		    toSubmitCSV.to_csv(OUTPUT_DIRECTORY+'Predictions/'+type(clf).__name__+'outputPrediction.csv', index=None)
