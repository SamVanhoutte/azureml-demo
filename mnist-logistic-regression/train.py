# Force latest version of arcus packages
import subprocess
import sys

# General references
import argparse
import os
import numpy as np
import pandas as pd
import joblib

# Add arcus references
from arcus.ml import dataframes as adf
from arcus.ml.timeseries import timeops
from arcus.ml.images import *
from arcus.ml.evaluation import classification as clev
from arcus.azureml.environment.aml_environment import AzureMLEnvironment
from arcus.azureml.experimenting.aml_trainer import AzureMLTrainer

# Add AzureML references
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
from azureml.core import VERSION

# This section enables to use the module code referenced in the repo
import os
import os.path
import sys
import time
from datetime import date

import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler


##########################################
### Parse arguments and prepare environment
##########################################

parser = argparse.ArgumentParser()

# If you want to parse arguments that get passed through the estimator, this can be done here
parser.add_argument('--c_value', type=float, dest='c_value', default=1, help='C-Value')
parser.add_argument('--solver', type=str, dest='solver', default='lbfgs', help='Solver type')
parser.add_argument('--class_weight', type=str, dest='class_weight', default='balanced', help='Class weight')
parser.add_argument('--multi_class', type=str, dest='multi_class', default='multinomial', help='Multi class')
parser.add_argument('--train_test_split_ratio', type=float, dest='train_test_split_ratio', default=0.3, help='Train test split ratio')

args, unknown = parser.parse_known_args()
c_value = args.c_value
solver = args.solver
class_weight = args.class_weight
multi_class = args.multi_class
train_test_split_ratio = args.train_test_split_ratio

# Load the environment from the Run context, so you can access any dataset
aml_environment = AzureMLEnvironment.CreateFromContext()
trainer = AzureMLTrainer.CreateFromContext()

if not os.path.exists('outputs'):
    os.makedirs('outputs')

##########################################
### Access datasets
##########################################

aml_df = aml_environment.load_tabular_dataset('mnist')
df = aml_df.copy()

##########################################
### Perform training
##########################################

# Load data 
y = df.label.values # The label is what we should predict = the y in the equation
X = np.asarray(df.drop(['label'],axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=0)

print(f'Train records: {len(X_train)}')
print(f'Test records: {len(X_test)}')
print(f'Input features: {X_train.shape[1]}')

# Build model 
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)



# Custom metrics tracking
# trainer._log_metrics('dice_coef_loss', list(fitted_model.history.history['dice_coef_loss'])[-1], description='')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight=class_weight, multi_class=multi_class, solver=solver, C=c_value)
model.fit(X_train, y_train)
trainer.evaluate_classifier(model, X_test, y_test, show_roc = False, upload_model = True)

##########################################
### Save model
##########################################
import pickle


filename = 'outputs/model.sav'
pickle.dump(model, open(filename, 'wb'))

print('Training finished')