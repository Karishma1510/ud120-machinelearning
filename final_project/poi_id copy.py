#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
# 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 
# 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi'] 

features_list = ['poi', 'exercised_stock_options', 'bonus', "fraction_to_poi", "loan_advances"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

count = 0
for key, value in data_dict.items():
    if data_dict[key]["poi"] == 1:
        count += 1
print("POI count", count)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("LOCKHART EUGENE E")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("WROBEL BRUCE")

### Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity
    if poi_messages == 'NaN':
        return 0
    elif all_messages == 'NaN' or all_messages == 0:
        return 0
    else:
        return (float(poi_messages/all_messages))

### Store to my_dataset for easy export below.
my_dataset = data_dict
for name in data_dict:
    data_point = data_dict[name]


    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi
    my_dataset[name]["fraction_to_poi"] = fraction_to_poi

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# plt.scatter(labels, features)
# plt.show()
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler().fit(features)
features_transformed = scaler.transform(features)



# Calculate chi2 for feature importance
chi2_res = chi2(features_transformed, labels)
# {
#  'exercised_stock_options': 6.845509335034564, 
#  'loan_advances': 6.688781738342222, 
#  'total_stock_value': 5.476610099286039, 
#  'bonus': 5.120754137086806, 
#  'salary': 3.0527867447897883, 
#  'total_payments': 2.784778839650515, 
#  'long_term_incentive': 2.538485033080888, 
#  'shared_receipt_with_poi': 2.4322198651432254, 
#  'other': 1.715950530799459, 
#  'director_fees': 1.501130853594847, 
#  'expenses': 1.4861033666636148, 
#  'from_poi_to_this_person': 1.370059292229467, 
#  'from_this_person_to_poi': 1.0008076418017091, 
#  'restricted_stock': 0.5895353494865818, 
#  'to_messages': 0.4363977688023856, 
#  'deferred_income': 0.3400992184059575, 
#  'from_messages': 0.06873854215131583, 
#  'deferral_payments': 0.060696606931361766, 
#  'restricted_stock_deferred': 0.0035067650332054516
# }

result = {}
for i in range(1,len(features_list)):
    result[features_list[i]] = chi2_res[0][i-1]
sorted_chi2_res = sorted(result.items(), key=lambda x:x[1], reverse=True)
print(dict(sorted_chi2_res))

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features_transformed, labels, test_size=0.3, random_state=42)

k = np.arange(10)+2
leaf = np.arange(30)+1
params_grid = [{'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'n_neighbors': k}]
tree_clf= GridSearchCV(KNeighborsClassifier(), param_grid=params_grid, scoring='f1', cv = 5)
tree_clf.fit(features_transformed, labels)
print(tree_clf.best_params_)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.metrics import accuracy_score, precision_score, recall_score

clf = GaussianNB().fit(features_train,labels_train)
pred = clf.predict(features_test)
print(accuracy_score(labels_test,pred), precision_score(labels_test, pred), recall_score(labels_test, pred))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)