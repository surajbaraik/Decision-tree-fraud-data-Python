# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:15:27 2020

@author: suraj baraik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fraud = pd.read_csv("C:\\Users\\suraj baraik\\Desktop\\Data Science\\Suraj\\New folder (12)\\Module 19 Decision Tree Random Forest\\Fraud_check.csv")
data = fraud.head()

fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"

fraud["income"].unique()
fraud["income"].value_counts()
### dropping the Taxable.Income columns
fraud = fraud.drop(['Taxable.Income'],axis=1)
fraud.rename(columns={"Marital.Status":"marital","City.Population":"population","Work.Experience":"workexp"},inplace=True)
fraud.isnull().sum()

## As i got an error during fitting the data for model building. I performed encoding using the below code.
## The error was "ValueError: could not convert string to float: 'NO'.
###You can't pass str to your model fit() method, so converting data type
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
    
features = fraud.iloc[:,0:5]
labels = pd.DataFrame(fraud.iloc[:,5])
fraud["income"].value_counts()

        
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.3, stratify=labels)

print(y_train["income"].value_counts())
print(y_test["income"].value_counts())
        
        
## We also use pd.factorize to convert the datatype, we  can use the below code.
#fraud["Undergrad"],_ = pd.factorize(fraud["Undergrad"])
#fraud["marital"],_ = pd.factorize(fraud["marital"])
#fraud["Urban"],_ = pd.factorize(fraud["Urban"])
##Converting the column names into the list format
colnames = list(fraud.columns)
predictors = colnames[:5]
target = colnames[5]

fraud.info()

###Splitting the data in train and test data
##One of the way to split the data
#fraud["is_train"] = np.random.uniform(0,1,len(fraud))<=0.70
#fraud["is_train"]
#train,test = fraud[fraud["is_train"]==True],fraud[fraud["is_train"]==False]

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
##Prediciton on train data 
pred_train = pd.DataFrame(model.predict(x_train))

### Finding the accuracy of train data
acc_train = accuracy_score(y_train,pred_train) #100%

##Confusion matrix for train data
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_train,pred_train))

##Prediction on test data
pred_test = pd.DataFrame(model.predict(x_test))

acc_test = accuracy_score(y_test,pred_test) ##68%

#confusion matrix for test data
cm_test = confusion_matrix(y_test,pred_test)

##Visualizing the decision trees

from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO


dot_data = StringIO()
export_graphviz(model, out_file = dot_data ,filled = True,rounded =True,feature_names = predictors,class_names = target, impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

###PDF file of Decision tree
graph.write_pdf('fraud.pdf')
##PNG file of Decision tree
graph.write_png('fraud.png')
