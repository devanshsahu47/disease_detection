import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

diabetes_dataset=pd.read_csv("C:/Users/HP/OneDrive/Documents/Disease_Detection/diabetes_disease/diabetes.csv")

X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

classifier=svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

input_data=(1,85,66,29,0,26.6,0.351,31)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=classifier.predict(input_data_reshaped)

if (prediction==0):
    print('The person is non-diabetic')
else:
    print('The person is diabetic')

filename='trained_diabetes_model.sav'
pickle.dump(classifier,open(filename,'wb'))