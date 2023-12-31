import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open('trained_diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('trained_heart_model.sav','rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=diabetes_model.predict(input_data_reshaped)

    if (prediction[0]==0):
        return 'The person is non-diabetic'
    else:
        return 'The person is diabetic'

def heart_prediction(input_data):
    input_data_numpy_array=np.asarray(input_data, dtype='float64')
    input_data_reshaped=input_data_numpy_array.reshape(1,-1)
    
    prediction=heart_disease_model.predict(input_data_reshaped)
    
    if (prediction[0]==0):
        return 'The person does not have a heart disease'
    else:
        return 'The person has a heart disease'

def main():
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               ['Diabetes Prediction',
                                'Heart Disease Prediction'],
                               icons=['activity','heart'],
                               default_index=0)
    
    if (selected == 'Diabetes Prediction'):
        st.title('Diabetes Prediction')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
        with col2:
            Glucose = st.text_input('Glucose Level')
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        with col2:
            Insulin = st.text_input('Insulin Level')
        with col3:
            BMI = st.text_input('BMI value')
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2:
            Age = st.text_input('Age of the Person')
        
        diab_diagnosis = ''
        
        if st.button('Diabetes Test Result'):
            diab_diagnosis = diabetes_prediction([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        st.success(diab_diagnosis)
    
    if (selected == 'Heart Disease Prediction'):
        st.title('Heart Disease Prediction')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')
        with col2:
            sex = st.text_input('Sex')
        with col3:
            cp = st.text_input('Chest Pain types')
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
        with col3:
            exang = st.text_input('Exercise Induced Angina')
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        heart_diagnosis = ''
        
        if st.button('Heart Disease Test Result'):
            heart_diagnosis = heart_prediction([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        st.success(heart_diagnosis)

if __name__=='__main__':
    main()
