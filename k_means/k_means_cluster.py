#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2",f3_name = "feature 3"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for ii, pp in enumerate(pred):
        ax.scatter(features[ii][0], features[ii][1],features[ii][2],color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                ax.scatter(features[ii][0], features[ii][1],features[ii][2],color="r", marker="*")
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_zlabel(f3_name)

    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
salary = []
stocks = []
for key,value in data_dict.items():
    if data_dict[key]["salary"] !="NaN":
        salary.append(data_dict[key]["salary"])
        
    if data_dict[key]["exercised_stock_options"] !="NaN":
        stocks.append(data_dict[key]["exercised_stock_options"])
from sklearn.preprocessing import MinMaxScaler

max_salary = float(max(salary))
min_salary = float(min(salary))
data = [[min_salary],[200000],[max_salary]]
print(MinMaxScaler().fit_transform(data))

print(max(stocks))
print(min(stocks))



### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2,feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
fig = plt.figure()
ax = plt.axes(projection='3d')
for f1, f2 ,f3 in finance_features:
    ax.scatter( f1, f2 ,f3)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2,n_init=10).fit(data)
pred = kmeans.predict(data)




### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_2,f3_name = feature_3)
   
except NameError:
    print("No predictions object named pred found, no clusters to plot")
