import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

past_data= pd.read_csv('loan_data.csv')



X= past_data.values[:,0: 13]
Y= past_data['not.fully.paid']

labelencoder= LabelEncoder()

onehotencoder= ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [1])], remainder= 'passthrough')
X=onehotencoder.fit_transform(X)

X=X[:, 1:]

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.4, random_state=4)

data_entropy= DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=4)
data_entropy.fit(X_train, Y_train)

y_pred= data_entropy.predict(X_test)

#to get accuracy of your prediction 
# print(f"Accuracy is {accuracy_score(Y_test, y_pred)*100}%")


