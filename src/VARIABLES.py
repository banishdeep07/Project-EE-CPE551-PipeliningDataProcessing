'''
Author: Github: banishdeep07
		Mail:	bojha@stevens.edu
'''

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
from xgboost import XGBRFRegressor


#Directories specification
ROOT_DIRECTORY = "../"
TRAIN_DIRECTORY = ROOT_DIRECTORY + "data/train/" # train directory
TEST_DIRECTORY = ROOT_DIRECTORY + "data/test/" # test directory that contains files
SAMPLE_SUBMISSION_PROVIDED = True #
SAMPLE_SUBMISSION_DIRECTORY = ROOT_DIRECTORY + "data/sample_submission/"
OUTPUT_DIRECTORY = ROOT_DIRECTORY + "/outputs/"
""" the outputs are: 
01.correlationAmongVariables.jpg
01.correlationAmongVariables.csv
00.trainData_X.pickle
00.trainData_Y.pickle
00.testData_X.pickle



plots: names as: model_name + 'Measure'+ .png
AUC curves for each model in case of classifier
Explained Variance and R2 measure for Regression

Outputs to be submitted: In folder Predictions/. names as: model_name+ 'outputPrediction.csv'

"""

#Some General Specification Controls
MULTIPLE_FILE_TRAIN = False #Join avoided
LABEL_AND_TRAIN_ON_SAME_TABLE = True # if not, specify LABELTABLE, and join key
LABEL_TABLE = False
JOIN_KEY = None
LABEL_IS_LAST_COLUMN_IN_TRAIN = False # Label on last column or not. if not, specify label column
LABEL_COLUMN = 'target' # to change this with name of label column
#LABEL_COLUMN = 'isFraud'
PROBLEM_TYPE = "Classification" # Other option being "Regression" among two oprions: {"Classification","Regression"}
USE_RANDOM_SEARCH = False # if to use RandomSearch, default False
USE_ENSEMBLE = False # if to use ensemble, default False


IMAGE_STRUCTURING = False
IMAGE_TRAIN_DIRECTORY = TRAIN_DIRECTORY + "Images/"
IMAGE_TEST_DIRECTORY = TEST_DIRECTORY + "Images/"
IMAGE_LABEL_ON_FLAT_FILE = True
IMAGE_LABEL_DIRECTORY = TRAIN_DIRECTORY
IMAGE_LABEL_FILE = "train.csv"
IMAGE_NAME_COLUMN = 'Image'
IMAGE_LABEL_COLUMN = 'Id'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


# Some Domain Specific Parameters
DISCRETE_THRESHOLD = 0.05 # if distinct value counts are less than this ratio, the value will be transformed to Categorical 
COLUMN_NA_THRESHOLD = 0.75 # if column contains greater than this ratio as NA, drop that column
ROW_TOTAL_NA_DROP_THRESHOLD = 0.05 # if no. of rows with any NA is less than this threshold, those rows are dropped


#CrossValidation Parameters
CROSS_VALIDATION_NUMBER = 10 # by default 10 folds cross validation


#Model Parameters
CUSTOMMODELS = False #False by default, use default set of models
'''
default models:
For Classification
LGBMClassifier : base model
MLPClassifier : base model
XGBoostClassifier : base model

For Regression
LGBMRegressor : base model
MLPRegressor : base model
XGBoostRegressor : base model


Imported models and can be used among these:
For Classification
Support Vector Classifier: from sklearn.svm import SVC
Gaussian Classifier: from sklearn.gaussian_process import GaussianProcessClassifier
Kernel RBF: from sklearn.gaussian_process.kernels import RBF
Decision Tree Classifier: from sklearn.tree import DecisionTreeClassifier
Random Forest and Adaptive Boosting Tree Classifier: from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
Gaussian Naive Bayes: from sklearn.naive_bayes import GaussianNB
k Nearest Neighbor Classifier: sklearn.neighbors import KNeighborsClassifier
Light GBM Classifier: from lightgbm import LGBMClassifier
Simple Neural Network Classifier: from sklearn.neural_network import MLPClassifier
XGBoost Classifier: from xgboost import XGBClassifier

For Regressor
Support Vector Regressor: from sklearn.svm import SVR
Gaussian Regressor: from sklearn.gaussian_process import GaussianProcessRegressor
Kernel RBF: from sklearn.gaussian_process.kernels import RBF
Decision Tree Regressor: from sklearn.tree import DecisionTreeRegressor
Random Forest and Adaptive Boosting Tree Regressor: from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
k Nearest Neighbor Regressor: from sklearn.neighbors import KNeighborsRegressor
Light GBM Regressor: from lightgbm import LGBMRegressor
Simple Neural Network Regressor: from sklearn.neural_network import MLPRegressor
XGBoost Regressor: from xgboost import XGBRFRegressor

Format should be in dictionary form like:
{model: {params}, model: {params}} Eg: models = {RandomForestClassifier:{'n_estimators':'warn', 'criterion':'gini', 'max_depth':None, 'min_samples_split':5, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0, 'max_features':'auto', 'max_leaf_nodes':None}, AdaBoostClassifier:{'n_estimators':100}}
'''

MODELS = {RandomForestClassifier:{'n_estimators':'warn', 'criterion':'gini', 'max_depth':None, 'min_samples_split':5, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0, 'max_features':'auto', 'max_leaf_nodes':None}, AdaBoostClassifier:{'n_estimators':100}, MLPClassifier:{}}
#If CUSTOMMODELS = True, define this dictionary to include desired models