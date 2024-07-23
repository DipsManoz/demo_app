import streamlit as st 
import joblib 
import sklearn
import pandas as pd 
import numpy as np 

model=joblib.load('iris_linear.pkl')
features=joblib.load('iris_cols.pkl')
# defining a function 
def pred(new_data): 
    x= model.predict(new_data)
    return x

# getting user data 
sepal_length=st.number_input('enter sepal lenght')
sepal_width=st.number_input('enter sepal width')
petal_length=st.number_input('enter petal length')
petal_width=st.number_input('enter petal width')

# creadte a predict button 
if st.button("predict"):
    new_data=pd.DataFrame(
        { 'sepal_length':[sepal_length],
           'sepal_width':[sepal_width],
           'petal_length':['petal_length'],
           'petal_width':['petal_width']
         }
    )
    
    p=pred(new_data[features])

    st.write(f"The predicted value is {p[0]}")