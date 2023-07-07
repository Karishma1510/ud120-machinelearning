#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

# how many total number of people in enron data
print(len(enron_data))

# How many features for each person
key = list(enron_data.keys())[0]
print(len(enron_data[key]))

# How many poi there in dataset
key = list(enron_data.keys())

print(key)
poi_list = []
for i in key:
    if enron_data[i]["poi"]==1:
        poi_list.append(i)
print(len(poi_list))

# What is the total value of the stock belonging to James Prentice?
print(enron_data["PRENTICE JAMES"]['total_stock_value'])

# How many email messages do we have from Wesley Colwell to persons of interest?
print(enron_data["COLWELL WESLEY"]['from_this_person_to_poi'])
    
# What’s the value of stock options exercised by Jeffrey K Skilling?
print(enron_data["SKILLING JEFFREY K"]['exercised_stock_options'])

#  How is it denoted when a feature doesn’t have a well-defined value?
# NaN


# How many folks in this dataset have a quantified salary? What about a known email address?
salary_quantify = []
for i in key:
    if enron_data[i]["salary"] != "NaN":
        salary_quantify.append(i)
print(len(salary_quantify))
email_address = []
for i in key:
    if enron_data[i]["email_address"] != "NaN":
        email_address.append(i)
print(len(email_address))

# How many people in the E+F dataset (as it currently exists) 
# have “NaN” for their total payments? What percentage of people 
# in the dataset as a whole is this?

total_payments_nan = []
for i in key:
    if enron_data[i]["total_payments"] == "NaN":
        total_payments_nan.append(i)
print(len(total_payments_nan))
print((21/146)*100)

# How many POIs in the E+F dataset have “NaN” for their 
# total payments? What percentage of POI’s as a whole is this?

total_poi = []
for i in key:
    if enron_data[i]["poi"]==1 and enron_data[i]["total_payments"] == "NaN" :
        total_poi.append(i)

a = len(total_poi)
print(a)
print((a/21)*100)


# If a machine learning algorithm were to use total_payments
#  as a feature, would you expect it to associate a “NaN” value 
# with POIs or non-POIs?
# Answer is non-POIs


# What is the new number of people of the dataset? What is the 
# new number of folks with “NaN” for total payments?
# Answer is number of people = 146+10 = 156 and
# folks with "NaN" for total payments = 21+10=31

# What is the new number of POI’s in the dataset? What is the new 
# number of POI’s with NaN for total_payments?
# total poi== =18+10 = 28 and 
# poi with NaN for total payments = 31




