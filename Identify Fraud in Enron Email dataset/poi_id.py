#!/usr/bin/python

import pprint
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../tools/")
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

################ Task 1: Select what features you'll use.######################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

### All potential features

email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = ['poi'] + email_features_list + financial_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Find NaN features for a given variable
def find_nan(data,variable):
    """ in a dictionary data, for the key variable, return the number of 
        non-NaN value. 
    """
    total_count = 0
    for k in data:
        if data[k][variable] == 'NaN':
            total_count = total_count + 1
    return total_count, 1.* total_count/len(data)

for f in features_list:
    print 'Features:'+ f 
    print 'number and percentage of missing values:',':',find_nan(data_dict,f)

########################### Task 2: Remove outliers ###########################

### Store to my_dataset for easy export below.
### Extract features and labels from dataset for local testing
my_dataset = data_dict

def exploration(data_dictionary):
    '''given a dictionary, print the number of keys and features, an 
    example data point, and the number and percentage of poi'''
    total_people = len(data_dictionary)
    print "number of people in the dataset:", total_people
    # print "the dataset information: ", data_dictionary
    total_keys = len(data_dictionary['METTS MARK'])
    print "poi is the label, number of all other features in the dataset is", 
    total_keys-1
    print "an example entry METTS MARK in the dataset: "
    pprint.pprint(data_dictionary['METTS MARK'])
    # count POI in the dataset
    total_poi = 0
    for k in data_dictionary:
        if data_dict[k]["poi"] == True:
            total_poi += 1
    print "number of poi in the dataset: ", total_poi
    print "percentage of poi in the dataset: ", 1.0*total_poi/total_people

### Explore the dataset
exploration(my_dataset) 

features_test = ["salary", "bonus"]

data_test = featureFormat(my_dataset, features_test)

for point in data_test:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#drop the outlier

my_dataset.pop("TOTAL", 0)

data_test = featureFormat(my_dataset, features_test)
for point in data_test:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

keys = []
for k in my_dataset:
    keys.append(k)

keys = sorted(keys)
print keys 

###'THE TRAVEL AGENCY IN THE PARK' in keys is not a person's name. so remove it.
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0 )
### Explore the dataset again
exploration(my_dataset) 

########################### Task 3: Create new feature(s) #####################

### Create new features "from_poi_to_this_person_ratio" and 
### "from_this_person_to_poi_ratio".

for name in my_dataset:
    if my_dataset[name]['from_poi_to_this_person'] != 'NaN':
        my_dataset[name]['from_poi_to_this_person_ratio'] = \
        1.0 * my_dataset[name]['from_poi_to_this_person']/my_dataset[name]['to_messages']
    else:
        my_dataset[name]['from_poi_to_this_person_ratio'] = 'NaN'
        
for name in my_dataset:        
    if my_dataset[name]['from_this_person_to_poi'] != 'NaN':
        my_dataset[name]['from_this_person_to_poi_ratio'] = \
        1.0 * my_dataset[name]['from_this_person_to_poi']/my_dataset[name]['from_messages']
    else:
        my_dataset[name]['from_this_person_to_poi_ratio'] = 'NaN'
        
### Add "from_poi_to_this_person_ratio" and "from_this_person_to_poi_ratio" into
### feature list.


features_list  = ['poi','salary', 'deferral_payments', 'loan_advances', 'bonus', 
                  'restricted_stock_deferred', 'deferred_income', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive', 
                  'restricted_stock', 'director_fees','total_stock_value', 
                  'to_messages',  'total_payments','from_poi_to_this_person', 
                  'from_messages','from_this_person_to_poi', 
                  'shared_receipt_with_poi', 'from_poi_to_this_person_ratio',
                  'from_this_person_to_poi_ratio']
### Since at start it is found that the values in loan_advances are all "NaN", 
### so the 'loan_advances' is removed from the list.
features_list.remove('loan_advances')
print features_list,len(features_list)

data = featureFormat(my_dataset, features_list)
poi, features = targetFeatureSplit(data)
 
### Split the data into train and test
features_train, features_test, labels_train, labels_test = train_test_split( 
                                features, poi, test_size=0.3, random_state=42)      

### Scale features via min-max
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.fit_transform(features_test)


###SelectKBest score

selector = SelectKBest()
selector.fit(rescaled_features_train, labels_train)
scores =  selector.scores_

### distribution of score in selectkbest
plt.hist(scores,bins = 19)
plt.title("SelectKBest score disctribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()

### Rank features based on scores
i = 0
features_score = []
while i < len(scores):
    features_score.append((features_list[i+1],scores[i]))
    i = i + 1
features_score = sorted(features_score, key = lambda x: x[1],reverse = True)
print features_score

### Final features were selected based on socres, features with scores > 3
### were included.
features_list = ['poi', 'exercised_stock_options', 'total_stock_value',
                  'from_this_person_to_poi_ratio', 'expenses','salary',
                  'deferred_income']

### Find NaN values in final features for a given variable

for feature in features_list :
    print 'number and percentage of missing values for', feature, ':',find_nan(my_dataset,feature)



########################## Task 4: Try a varity of classifiers#################
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

### Potential classifiers

clf_names = ["Naive Bayes","SVM RBF","Decision Tree","KMeans",
             "Random Forest"]

clfs = [GaussianNB(),SVC(kernel="rbf",gamma=0.1,C = 1000), DecisionTreeClassifier(),
        KMeans(n_clusters=2, tol=0.001),
        RandomForestClassifier(max_depth = 5,max_features = 'sqrt', 
                               n_estimators = 10, random_state = 42)]


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2:{:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


### Compare 5 classifiers
for name, clf in zip(clf_names, clfs):
    print '#############################################################'
    print '                          ' + name
    print '#############################################################'
    test_classifier(clf,my_dataset,features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

# Based on previous results, the Random Forest is the best classifier, so it was 
# chosed to be further tuned. 

#Two parameters to be tuned.

max_features = ['sqrt','log2',None]
max_depth = [5,10,20,40,None]

### To ease the reviewing process, since try all these parameters may cause more
### than 2 hours, the best parameters are given below.

for f in max_features:
    for d in max_depth:
        clf = DecisionTreeClassifier(max_features=f, max_depth=d)
        print 'max_features:',f,'max_depth',d
        test_classifier(clf,my_dataset,features_list)


### Final classifier
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
            max_features='sqrt', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

###Test final clf with new features
test_classifier(clf,my_dataset,features_list)

### Test final clf without the new feature
nonew_features = ['poi', 'exercised_stock_options', 'total_stock_value',
                  'expenses','salary','deferred_income']

test_classifier(clf,my_dataset,nonew_features)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#Dump1
print features_list
dump_classifier_and_data(clf, my_dataset, features_list)

### Double-check dumped variables:
    
CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def load_classifier_and_data():
    clf = pickle.load(open(CLF_PICKLE_FILENAME, "r") )
    dataset = pickle.load(open(DATASET_PICKLE_FILENAME, "r") )
    feature_list = pickle.load(open(FEATURE_LIST_FILENAME, "r"))
    return clf, dataset, feature_list

clf, dataset, feature_list = load_classifier_and_data()

print 'pickled clf: ', clf
print 'pickled features list: ' , feature_list

#Dump2

print nonew_features
dump_classifier_and_data(clf, my_dataset, nonew_features)

### Double-check dumped variables:
clf, dataset, feature_list = load_classifier_and_data()

print 'pickled clf: ', clf
print 'pickled features list: ' , feature_list