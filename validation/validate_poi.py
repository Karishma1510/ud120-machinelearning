#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys

from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree




data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state = 42)
print(labels)


### it's all yours from here forward! 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.score(features,labels))


clf = clf.fit(features_train,labels_train)
print(clf.score(features_test,labels_test))




