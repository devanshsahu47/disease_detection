import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

heart_data = pd.read_csv('C:/Users/HP/OneDrive/Documents/Disease_Detection/heart_disease/heart_disease_data.csv')

X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

model=LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

if (prediction==0):
    print('The person does not have a heart disease')
else:
    print('The person has a heart disease')

filename='trained_heart_model.sav'
pickle.dump(model,open(filename,'wb'))