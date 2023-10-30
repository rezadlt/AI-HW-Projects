#!/usr/bin/env python
# coding: utf-8

# In[241]:


import os
from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mac/Desktop/AI/diabet/diabetes.csv',header = 0, encoding = 'ascii', engine = 'python')
df.head()


# In[242]:


filename = '/Users/mac/Desktop/AI/diabet/diabetes.csv'
data = pd.read_csv(filename)
print (data.describe())


# In[243]:


# Define the calculate entropy function
def calculate_entropy(df_label):
    classes,class_counts = np.unique(df_label,return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts)) 
                        for i in range(len(classes))])
    return entropy_value


# In[244]:


# Define the calculate information gain function
def calculate_information_gain(dataset,feature,label): 
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])   
    values,feat_counts= np.unique(dataset[feature],return_counts=True)
    
    # Calculate the weighted feature entropy                                # Call the calculate_entropy function
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]
                              ==values[i]).dropna()[label]) for i in range(len(values))])    
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain


# In[245]:


features = df.columns[:-1]
label = 'diabetes'
parent = None
features


# In[246]:


df.hist(bins = 50, figsize = (20, 15))
plt.show()


# In[247]:


median_bmi = df['BMI'].median()
mean_bmi = df['BMI'].mean()
print("The median BMI is :",median_bmi)
print("The mean BMI is :",mean_bmi)


# In[248]:


median_bp = df['BloodPressure'].median()
mean_bp = df['BloodPressure'].mean()
print('Median Blood Pressure is:',median_bp)
print('Mean Blood Preddure is :',mean_bp)


# In[249]:


median_insulin = df['Insulin'].median()
mean_insulin = df['Insulin'].mean()
print('Median Insulin is:',median_insulin)
print('Mean Insulin is :',mean_insulin)


# In[250]:


median_skin = df['SkinThickness'].median()
mean_skin = df['SkinThickness'].mean()
print('Median Skin Thickness is:',median_skin)
print('Mean SkinThickness is :',mean_skin)


# In[251]:


median_preg = df['Pregnancies'].median()
mean_preg = df['Pregnancies'].mean()
print('Median Pregnancy is:',median_preg)
print('Mean Pregnancy is :',mean_preg)


# In[252]:


median_age = df['Age'].median()
mean_age = df['Age'].mean()
print('Median Age is:',median_age)
print('Mean Age is :',mean_age)


# In[253]:


median_glucose = df['Glucose'].median()
mean_glucose = df['Glucose'].mean()
print('Median Glucose is:',median_glucose)
print('Mean Glucose is :',mean_glucose)


# In[254]:


median_DiabetesPedigreeFunction = df['DiabetesPedigreeFunction'].median()
mean_DiabetesPedigreeFunction = df['DiabetesPedigreeFunction'].mean()
print('Median DiabetesPedigreeFunction is:',median_DiabetesPedigreeFunction)
print('Mean DiabetesPedigreeFunction is :',mean_DiabetesPedigreeFunction)


# In[255]:


df['BMI'] = df['BMI'].replace(
    to_replace=0, value = median_bmi)
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(
    to_replace=0, value = median_DiabetesPedigreeFunction)
df['Glucose'] = df['Glucose'].replace(
    to_replace=0, value = median_glucose)
df['Age'] = df['Age'].replace(
    to_replace=0, value = median_age)
df['Pregnancies'] = df['Pregnancies'].replace(
    to_replace=0, value = median_preg)
df['SkinThickness'] = df['SkinThickness'].replace(
    to_replace=0, value = median_skin)
df['Insulin'] = df['Insulin'].replace(
    to_replace=0, value = median_insulin)
df['BloodPressure'] = df['BloodPressure'].replace(
    to_replace=0, value = median_bp)


# In[256]:


train, test = train_test_split(df, test_size = 0.4, random_state=30)
target = train["Outcome"]
feature = train[train.columns[0:8]]
feat_names = train.columns[0:8]
target_classes = ['0','1'] 
print(test)


# In[257]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=21)
neigh.fit(feature,target)
TrueDecisionpredicted = neigh.predict(test_input)
print(metrics.classification_report(expected, TrueDecisionpredicted))
print(metrics.confusion_matrix(expected, knnpredicted))
print("TrueDecision accuracy: ",neigh.score(test_input,expected))
TrueDecisionscore=neigh.score(test_input,expected)


# In[258]:


import graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
print(dot_data)
Image(graph.create_png())

