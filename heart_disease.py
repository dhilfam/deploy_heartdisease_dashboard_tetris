# Prepare the library
from typing import Any 
import streamlit as st
import pandas as pd 

import pickle
import time
from PIL import Image

st.set_page_config(page_title="Dashboard Fadhil", 
                   layout="wide")

# Introduction
st.write("""
        # Hello! Welcome to [M.Fadhil](https://www.linkedin.com/in/muhammad-fadhil-18aba6259/)'s Machine Learning Dashboard
        """)

add_selectitem = st.sidebar.selectbox("Want to open about?", ("Heart Disease!",))

st.write("""
        # This app predicts the **Heart Disease**.
        
        Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)

# Collects user input features into dataframe
st.sidebar.header('User Input Features:')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Typical angina"
            elif cp == 2.0:
                wcp = "Atypical angina"
            elif cp == 3.0:
                wcp = "Non angina"
            else:
                wcp = "Asymptomatic"
            st.sidebar.write("Type of Chest pain : ", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Slope of the peak exercise ST segment", 0, 2, 1)
            oldpeak = st.sidebar.slider("ST depression induced", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Result of thallium test", 1, 3, 1)
            sex = st.sidebar.selectbox("Sex", ('Female', 'Male'))
            if sex == "Female":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Age", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

#List of image file names
image_files = ['jantung.jpeg', 'jantung2.jpeg']
st.write("""
__Heart disease refers to a group of conditions that can affect the heart's functioning. This includes coronary artery disease, heart failure, arrhythmias, and many other conditions. 
Heart disease can be life-threatening and is often associated with risk factors such as high blood pressure, high cholesterol levels, poor dietary habits, smoking, lack of physical activity, and genetic factors.__
""")

#Desired image size in pixels
desired_width = 160
desired_height = 160

#Display resized images using Streamlit's layout options
col1, col2, col3, col4 = st.columns(4)

for idx, image_file in enumerate(image_files):
        img = Image.open(image_file)
        
#Resized the image to the desired size
        resized_img = img.resize((desired_width, desired_height))

#Display resized images in respective columns
        if idx == 0:
                col1.image(resized_img, caption=image_file, use_column_width=True)
        else:
                col2.image(resized_img, caption=image_file, use_column_width=True)
    
#Loading images
heartdisease= Image.open('jantungsakit.jpeg')
strongheart =Image.open('jantungsehat.jpeg')

loaded_model = None
output = " "

if st.sidebar.button('Click Here To Predict'):
    df = input_df
    st.write(df)
    with open("best_model_rfs.pkl", 'rb') as file:  
        loaded_model = pickle.load(file)

if loaded_model is not None:        
        prediction = loaded_model.predict(df)
        if (prediction == 0).any():
            result = 'Negative Heart Disease'
        else:
            result = 'Positive Heart Disease'
        
        output = str(result)

        st.subheader('Prediction Result: ')
        
        with st.spinner('Wait for it...'):
                time.sleep(4)
          
        st.success(f"Prediction of this app is {output}")
        
        if (prediction == 0).any():
                st.image(strongheart)
        else:
                st.image(heartdisease)