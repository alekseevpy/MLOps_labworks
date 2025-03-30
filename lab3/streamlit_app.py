import io
import streamlit as st
from PIL import Image
import numpy as np
import pickle

fields = {'radius1':-0.46649743,
'texture1':-0.13728933,
'perimeter1':-0.44421138,
'area1':-0.48646498,
'smoothness1':0.28085007,
'compactness1':0.04160589,
'concavity1':-0.11146496,
'concave_points1':-0.26486866,
'symmetry1':0.41524141,
'fractal_dimension1':0.13513744,
'radius2':-0.02091509,
'texture2':-0.29323907,
'perimeter2':-0.17460869,
'area2':-0.2072995,
'smoothness2':-0.01181432,
'compactness2':-0.35108921,
'concavity2':-0.1810535,
'concave_points2':-0.24238831,
'symmetry2':-0.33731758,
'fractal_dimension2':-0.0842133,
'radius3':-0.2632354,
'texture3':-0.14784208,
'perimeter3':-0.33154752,
'area3':-0.35109337,
'smoothness3':0.48001942,
'compactness3':-0.09649594,
'concavity3':-0.03583041,
'concave_points3':-0.19435087,
'symmetry3':0.17275669,
'fractal_dimension3':0.20372995}

input_values = {'radius1': 0.0,
'texture1': 0.0,
'perimeter1': 0.0,
'area1': 0.0,
'smoothness1': 0.0,
'compactness1': 0.0,
'concavity1': 0.0,
'concave_points1': 0.0,
'symmetry1': 0.0,
'fractal_dimension1': 0.0,
'radius2': 0.0,
'texture2': 0.0,
'perimeter2': 0.0,
'area2': 0.0,
'smoothness2': 0.0,
'compactness2': 0.0,
'concavity2': 0.0,
'concave_points2': 0.0,
'symmetry2': 0.0,
'fractal_dimension2': 0.0,
'radius3': 0.0,
'texture3': 0.0,
'perimeter3': 0.0,
'area3': 0.0,
'smoothness3': 0.0,
'compactness3': 0.0,
'concavity3': 0.0,
'concave_points3': 0.0,
'symmetry3': 0.0,
'fractal_dimension3': 0.0}

@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()
st.title('Предсказание наличия рака молочной железы')
st.write('Ниже перечислены 30 признаков, используемых в наборе данных sklearn.datasets.load_breast_cancer')
st.write('Оригинальный набор данных взят с https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic')
for label, example_value in fields.items():
    input_values[label] = st.number_input(label=label, value=example_value, step=0.00000001, format="%0.8f")
result = st.button('Предсказать результат')
if result:
    st.write(input_values)
    X = np.reshape(list(input_values.values()), (1, -1))
    y = model.predict_proba(X)
    st.write(y)