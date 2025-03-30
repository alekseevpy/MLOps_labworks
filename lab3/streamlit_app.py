import io
import streamlit as st
from PIL import Image
import numpy as np
import pickle


@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def take_text():
    inputted_text = st.text_input(label='Внесите строку с данными в формате датасета load_breast_cancer')

    if inputted_text is not None:
        print(inputted_text)
    else:
        return None

    return inputted_text


model = load_model()

st.title('Предсказание наличия рака молочной железы')
st.write('[[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01,  1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02,  6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01,  1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01,  4.601e-01 1.189e-01]]')
x = take_text()
result = st.button('Предсказать результат')
if result:
    y = model.predict(x)
    st.write(y)