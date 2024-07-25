#!/usr/bin/env python
# coding: utf-8

# # Lab 8: Implement Your Machine Learning Project Plan

# In this lab assignment, you will implement the machine learning project plan you created in the written assignment. You will:
# 
# 1. Load your data set and save it to a Pandas DataFrame.
# 2. Perform exploratory data analysis on your data to determine which feature engineering and data preparation techniques you will use.
# 3. Prepare your data for your model and create features and a label.
# 4. Fit your model to the training data and evaluate your model.
# 5. Improve your model by performing model selection and/or feature selection techniques to find best model for your problem.

# ### Import Packages
# 
# Before you get started, import a few packages.

# In[1]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# <b>Task:</b> In the code cell below, import additional packages that you have used in this course that you will need for this task.

# In[ ]:





# In[2]:


from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV


# ## Part 1: Load the Data Set
# 
# 
# You have chosen to work with one of four data sets. The data sets are located in a folder named "data." The file names of the three data sets are as follows:
# 
# * The "adult" data set that contains Census information from 1994 is located in file `adultData.csv`
# * The airbnb NYC "listings" data set is located in file  `airbnbListingsData.csv`
# * The World Happiness Report (WHR) data set is located in file `WHR2018Chapter2OnlineData.csv`
# * The book review data set is located in file `bookReviewsData.csv`
# 
# 
# 
# <b>Task:</b> In the code cell below, use the same method you have been using to load your data using `pd.read_csv()` and save it to DataFrame `df`.

# In[3]:


adultDataSet_filename = os.path.join(os.getcwd(), "data", "adultData.csv")
df = pd.read_csv(adultDataSet_filename)

column_data_types = df.dtypes
print(column_data_types)


# ## Part 2: Exploratory Data Analysis
# 
# The next step is to inspect and analyze your data set with your machine learning problem and project plan in mind. 
# 
# This step will help you determine data preparation and feature engineering techniques you will need to apply to your data to build a balanced modeling data set for your problem and model. These data preparation techniques may include:
# * addressing missingness, such as replacing missing values with means
# * renaming features and labels
# * finding and replacing outliers
# * performing winsorization if needed
# * performing one-hot encoding on categorical features
# * performing vectorization for an NLP problem
# * addressing class imbalance in your data sample to promote fair AI
# 
# 
# Think of the different techniques you have used to inspect and analyze your data in this course. These include using Pandas to apply data filters, using the Pandas `describe()` method to get insight into key statistics for each column, using the Pandas `dtypes` property to inspect the data type of each column, and using Matplotlib and Seaborn to detect outliers and visualize relationships between features and labels. If you are working on a classification problem, use techniques you have learned to determine if there is class imbalance.
# 
# 
# <b>Task</b>: Use the techniques you have learned in this course to inspect and analyze your data. 
# 
# <b>Note</b>: You can add code cells if needed by going to the <b>Insert</b> menu and clicking on <b>Insert Cell Below</b> in the drop-drown menu.

# In[4]:


# Address missingness
df['age'].fillna(df['age'].median(), inplace=True)
df['hours-per-week'].fillna(df['hours-per-week'].median(), inplace=True)
categorical_cols = ['workclass', 'occupation', 'native-country']
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

# Addressing Outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
z_scores = np.abs(zscore(df[numeric_cols]))
threshold = 3
outliers = np.where(z_scores > threshold)
for col in numeric_cols:
    df[col] = winsorize(df[col], limits=[0.05, 0.05])

    
df.columns


# In[5]:


# Now very important, in my last assignment, It was clear that data on different non-white races were very imbalanced, so lets try and fix that
# Doing it for race
majority_class = df[df['race'] == 'White']
minority_classes = df[df['race'] != 'White']
print("Before Balancing")
class_distribution = df['race'].value_counts()
print("Class Distribution:")
print(class_distribution)
upsampled_minority_classes = []
for minority_race in minority_classes['race'].unique():
    minority_class = minority_classes[minority_classes['race'] == minority_race]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    upsampled_minority_classes.append(minority_upsampled)
df = pd.concat([majority_class] + upsampled_minority_classes)
df = df.sample(frac=1, random_state=42)  # Shuffle the balanced dataset
print("After Balancing")
class_distribution_after = df['race'].value_counts()
print("Class Distribution:")
print(class_distribution_after)


# Doing it for Gender
majority_class_gender = df[df['sex_selfID'] == 'Non-Female']
minority_class_gender = df[df['sex_selfID'] == 'Female']
print("Before Balancing")
class_distribution_gender = df['sex_selfID'].value_counts()
print("Class Distribution:")
print(class_distribution_gender)
# Upsample the minority class to address class imbalance for gender
minority_upsampled_gender = resample(minority_class_gender, replace=True, n_samples=len(majority_class_gender), random_state=42)
df = pd.concat([majority_class_gender, minority_upsampled_gender])
df = df.sample(frac=1, random_state=42)  # Shuffle the balanced dataset
print("After Balancing")
class_distribution_after_gender = df['sex_selfID'].value_counts()
print("Class Distribution:")
print(class_distribution_after_gender)
print(df.columns)  # Print the columns of the balanced DataFrame


# In[6]:




# One-hot encode categorical data, very important for this prompt
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'sex_selfID', 'native-country', 'income_binary']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df_encoded
df.columns





# In[ ]:





# ## Part 3: Implement Your Project Plan
# 
# <b>Task:</b> Use the rest of this notebook to carry out your project plan. You will:
# 
# 1. Prepare your data for your model and create features and a label.
# 2. Fit your model to the training data and evaluate your model.
# 3. Improve your model by performing model selection and/or feature selection techniques to find best model for your problem.
# 
# 
# Add code cells below and populate the notebook with commentary, code, analyses, results, and figures as you see fit.

# In[7]:


# Load your DataFrame as 'df'
# Prepare data
X = df.drop('race', axis=1)  # Features
y = df['race']  # Label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)


# In[8]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[11]:



#Find Important features
#1. Obtain "feature importance" scores from the model object and save the array to the variable 
# 'feature_imp'

feature_imp = model.feature_importances_

#2. Create a Pandas DataFrame with a list of all features and their scores. 
# Save the result to the variable 'df_features'

df_features = pd.DataFrame({
    'name': X_train.columns.values,
    'imp': feature_imp
})

#3. Sort df_features in descending order and
# save the result to the variable 'df_sorted'

df_sorted = df_features.sort_values('imp',  ascending=False)

#4. Obtain the top 5 sorted feature names and save the result to list 'top_five' 

top_five = list(df_sorted['name'].iloc[:8])
print('Top five features: {0}'.format(top_five))


# In[13]:


fig, ax = plt.subplots()
ax.bar(np.arange(8), sorted(model.feature_importances_, reverse=True)[:8], width = 0.35)
ax.set_xticks(np.arange(8))
ax.set_xticklabels(top_five, rotation = 90)
plt.title('Feature importance from DT')
ax.set_ylabel('Normalized importance')


# In[ ]:



# Evaluate the initial model
initial_accuracy = model.score(X_test, y_test)
print(f"Initial Model Accuracy: {initial_accuracy:.2f}")

# Perform feature selection
rfe = RFE(model, n_features_to_select=10)  # Choose appropriate number of features
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]

# Fine-tune hyperparameters
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the final model
final_accuracy = best_model.score(X_test, y_test)
print(f"Final Model Accuracy: {final_accuracy:.2f}")


# In[ ]:


from sklearn.utils.multiclass import unique_labels

y_pred_prob = model.predict_proba(X_test)[:, 1]
n_classes = len(unique_labels(y_test))

fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=n_classes-1)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Binned ROC Curve')
plt.show()


# In[ ]:


Conclusion: 
    
    When building this model. I think I got a really good sense of the steps and what needs to be done in each area. 
    I followed the normal procedure when setting up the model. The data cleaning was easy but tedious. The first issue I 
    encountered was when I looked at distribution and bias. There were a lot on "White" race records and not much of any other
    catagory. The dataset got really large after that so I understand why the processing time was so long. The final steps were a
    little more confusing for me. I chose to train a Random Forest Classification model to get my predictions and it seemed to 
    perform well. I did a little bit more analysis by looking at the Accuracy, Plot Graph and Feature importance. However I was not 
    100% sure where to go when it came to optimizing models and using multiples and analying all the specific things. 
    
    When I got the accurraccy it seemed too high. I think my big conclusion is that there isnt enough descriptive data to accurately 
    take advantage of the model the way I wanted. I also think I am unclear on what happens after one-hot encoding with getting the 
    binary features to be read correctly by the model. I needed to have spent more time understanding that better so the later steps 
    could have been less confusing. I Am choosing to end it at this point because I feel I have performed some essential steps well. I
    Hope to poke and play around with it more once the stressors of moving & adjusting back to school have taken place. 
    
    


# In[ ]:





# In[ ]:




