
# coding: utf-8

# In[1]:


#!/usr/bin/python
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

sys.path.append("../tools/")

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split


# In[2]:


''' Miscellanious functions '''

def computeFraction(salary, bonus):
    """ Computing how much EXTRA salaries you will receive in bonus """    
    invalid_data = salary == 'NaN' or bonus == 'NaN'
    fraction = 0. if invalid_data else float(bonus) / float(salary)
    return fraction

def add_bonus_by_salary(data_dict):
    """ I'm creating a new feature merging mixing salary and bonus"""
    submit_dict = {}
    for name in data_dict:
        data_point = data_dict[name]        
        salary = data_point["salary"]
        bonus = data_point["bonus"]
        bonus_fraction = computeFraction(salary,bonus)        
        data_point["bonus_extra_salary"] = bonus_fraction   

def clean_observed_outliers(data_dict,key):
    data_dict.pop(key,0)    
    
def show_metrics(method,pred, labels_test, importances = None):    
    ac = accuracy_score(pred, labels_test)
    pr = precision_score(labels_test, pred, average="binary")
    re = recall_score(labels_test, pred, average="binary")
    f1 = f1_score(labels_test, pred, average="binary")    
    print "METHOD:", method
    print "\t","Features used:", features_list
    print "\t","Accuracy:",ac
    print "\t","Precision:",pr
    print "\t","Recall:",re
    print "\t","F1:",f1
    if (importances is not None):
        print "Importances:"
        i = 0    
        for imp in importances:    
            if imp > 0 and i<len(features_list)-1:                    
                i = i + 1                 
                print "\t",features_list[i],":",imp                   
    
def make_meshgrid(x, y, h=0.1):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """            
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_classifier_relation(ax, features_train,labels_train,x_name,y_name,model,h):
    
    # Take two features and plot the predicted classification
    X = pd.DataFrame(features_train, columns=features_list[1:])[[x_name,y_name]]
    y = pd.DataFrame(labels_train)[0]
    
    
    if model == 'GaussianNB':
        estimators = [('Feature Selection', PCA(svd_solver='randomized',n_components=2)),
                      ('Classification', GaussianNB())]
    elif model == 'DecisionTreeClassifier':
        estimators = [('Classification', DecisionTreeClassifier(min_samples_split=2, max_depth=15))]
    else:
        estimators = [('Feature Scaling', MinMaxScaler()),
                      ('Feature Selection', PCA(svd_solver='randomized',n_components=2)),
                      ('Classification', SVC(C=2450, gamma=5.7, kernel='rbf'))]
        
    clf = Pipeline(estimators)
    clf.fit(X,y)
    
    X0, X1 = X[x_name], X[y_name]
    xx, yy = make_meshgrid(X0, X1,h)
           
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


# In[3]:


from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',                    
                 'bonus_extra_salary',                 
                 'shared_receipt_with_poi',                 
                 'deferred_income',
                 'exercised_stock_options',                 
                 'long_term_incentive',
                 'bonus',
                 'salary',
                 'deferral_payments'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
clean_observed_outliers(data_dict,'TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
add_bonus_by_salary(my_dataset) #Adding new basic feature Fraction of extra salaries from bonus to and from POI

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[4]:


from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif

### Used feature selection thru SelectBestK and SelectPercentile methods.
### I used f_classif since I used features and labes and classification models.

X = features
y = labels
K = 5
P = 50

print "ORIGINAL FEATURES:"
print features_list[1:]
print
print "INTELLIGENT FEATURE SELECTION:"
selector1 = SelectPercentile(f_classif,percentile=P)
selector1.fit(X, y)
mask1 = selector1.get_support()
new_features1 = []
for bool, feature in zip(mask1,features_list[1:]):
    if bool:
        new_features1.append(feature)
print "SelectPercentile:(percentaje=50)", new_features1

selector2 = SelectKBest(f_classif,k=K)
selector2.fit(X, y)
mask2 = selector2.get_support()
new_features2 = []
for bool, feature in zip(mask2, features_list[1:]):
    if bool:
        new_features2.append(feature)
print "SelectKBest(k=5):", new_features2


# In[5]:


features_list = ['poi'] + new_features2

# We then store the split instance into cv and use it in our GridSearchCV.
cv = StratifiedShuffleSplit(n_splits=10,random_state=42)
parameters = [{'reduce_dim__n_components': [2, 3, 4,5]}]
pipe = Pipeline([('reduce_dim',PCA(svd_solver='randomized')),
                 ('classification', GaussianNB())])
grid = GridSearchCV(pipe, param_grid=parameters, cv = cv, scoring='average_precision').fit(features, labels)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))


# In[6]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#I switched to a basic train_test_split to improve validation time.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf1 = Pipeline([('reduce_dim',PCA(svd_solver='randomized', n_components=5)),
                 ('classification', GaussianNB())])

clf1.fit(features_train, labels_train)

print "Explained variance ratio pc:", clf1.named_steps['reduce_dim'].explained_variance_ratio_
print "First pc:", clf1.named_steps['reduce_dim'].components_[0]
print "Second pc:",clf1.named_steps['reduce_dim'].components_[1]

pred1 = clf1.predict(features_test) 
show_metrics("GaussianNB",pred1,labels_test)


# In[7]:


#I switched to a basic train_test_split to improve validation time.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf2 = DecisionTreeClassifier(min_samples_split=2)
clf2.fit(features_train, labels_train)

pred2 = clf2.predict(features_test) 
show_metrics("DecisionTreeClassifier",pred2,labels_test,clf2.feature_importances_)


# In[8]:


#I switched to a basic train_test_split to improve validation time.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Scaling features previous to dimensionality reduction using PCA
clf3 = Pipeline([('feature_scaling',MinMaxScaler()),
                 ('reduce_dim', PCA(svd_solver='randomized', n_components=2)),
                 ('classification', SVC(kernel='rbf',C=2450,gamma=5.7))]).fit(features_train, labels_train)

pred3 = clf3.predict(features_test)
show_metrics("SVC",pred3,labels_test)


# In[9]:


clf = clf1
pred = pred1


# In[10]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# In[11]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)


# In[12]:


# Plotting Naive bayes desicion boudary
'''
f, ax1 = plt.subplots(2, 3,figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plot_classifier_relation(ax1[0, 0], features_train,labels_train,features_list[1],features_list[2],'GaussianNB',0.1)
plot_classifier_relation(ax1[0, 1], features_train,labels_train,features_list[1],features_list[3],'GaussianNB',10)
plot_classifier_relation(ax1[0, 2], features_train,labels_train,features_list[1],features_list[4],'GaussianNB',10)
plot_classifier_relation(ax1[1, 0], features_train,labels_train,features_list[2],features_list[3],'GaussianNB',100)
plot_classifier_relation(ax1[1, 1], features_train,labels_train,features_list[2],features_list[4],'GaussianNB',100)
plot_classifier_relation(ax1[1, 2], features_train,labels_train,features_list[3],features_list[4],'GaussianNB',5000)
plt.show()
'''


# In[13]:


# Plotting Decision Tree Classifier
'''
f, ax2 = plt.subplots(2, 3,figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plot_classifier_relation(ax2[0, 0], features_train,labels_train,features_list[1],features_list[2],'DecisionTreeClassifier',0.1)
plot_classifier_relation(ax2[0, 1], features_train,labels_train,features_list[1],features_list[3],'DecisionTreeClassifier',5)
plot_classifier_relation(ax2[0, 2], features_train,labels_train,features_list[1],features_list[4],'DecisionTreeClassifier',5)
plot_classifier_relation(ax2[1, 0], features_train,labels_train,features_list[2],features_list[3],'DecisionTreeClassifier',100)
plot_classifier_relation(ax2[1, 1], features_train,labels_train,features_list[2],features_list[4],'DecisionTreeClassifier',100)
plot_classifier_relation(ax2[1, 2], features_train,labels_train,features_list[3],features_list[4],'DecisionTreeClassifier',5000)
plt.show()
'''


# In[14]:


# Plotting Naive bayes desicion boudary
'''
f, ax3 = plt.subplots(2, 3,figsize=(10,5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plot_classifier_relation(ax3[0, 0], features_train,labels_train,features_list[1],features_list[2],'SVC',0.5)
plot_classifier_relation(ax3[0, 1], features_train,labels_train,features_list[1],features_list[3],'SVC',10)
plot_classifier_relation(ax3[0, 2], features_train,labels_train,features_list[1],features_list[4],'SVC',10)
plot_classifier_relation(ax3[1, 0], features_train,labels_train,features_list[2],features_list[3],'SVC',100)
plot_classifier_relation(ax3[1, 1], features_train,labels_train,features_list[2],features_list[4],'SVC',100)
plot_classifier_relation(ax3[1, 2], features_train,labels_train,features_list[3],features_list[4],'SVC',50000)
plt.show()
'''

