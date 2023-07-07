#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
import numpy as np


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL")

data = featureFormat(data_dict, features)



### your code below
features_train,features_target = targetFeatureSplit(data)
print(features_train)
features_train = np.reshape(features_train,(len(features_train),1))
features_target = np.reshape(features_target,(len(features_target),1))

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg = reg.fit(features_train,features_target)




plt.scatter(features_train,features_target)
plt.show()



