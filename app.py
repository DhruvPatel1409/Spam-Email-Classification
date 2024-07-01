import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

vector_form = pickle.load(open('vector.pkl', 'rb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))

vectorization = TfidfVectorizer()

def predict_fake_mail(mail):
    input_data = [mail]
    transformed_input = vector_form.transform(input_data)
    prediction = loaded_model.predict(transformed_input)
    return prediction

st.title("Spam E-Mail Classification App")

title = st.text_input("Enter the mail:")

if st.button("Predict"):
    if title:
        prediction = predict_fake_mail(title)

        if prediction[0] == 0:
            st.success("Not Spam Detected!")
        else:
            st.error("Spam Detected!")
    else:
        st.warning("Please enter a title for prediction.")