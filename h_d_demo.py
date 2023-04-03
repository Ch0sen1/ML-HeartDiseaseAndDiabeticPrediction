"""
To run this app, in your terminal:
> streamlit run streamlit_demo.py

Source: https://is.gd/SobJvL
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading model 
clf = joblib.load('./model/heart-model.joblib')

# Create title and sidebar
st.title("HeartDisease & Diabetic Prediction")
st.sidebar.title("Features")

# Loading images
strongHeart = Image.open('strongHeart.jpg')
weakHeart = Image.open('weakHeart.jpg')


para_list = ['BMI', 'Smoking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
para_input = []
para_default_values=['20', '1', '0', '0', '0.833333','0','0','1','0','1','0.75','0','0','0']
res = []


for parameter, parameter_df in zip(para_list, para_default_values):
	values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=1.0, step=0.1)
	res.append(values)
 
input_va = pd.DataFrame([res],columns=para_list,dtype=float)
st.write('\n\n')


# Button that triggers the actual prediction
if st.button("Click Here to Classify"):
	prediction = clf.predict(input_va)
	if prediction == 0:
		st.image(strongHeart)
	elif prediction == 1:
		st.image(weakHeart)