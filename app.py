# app.py
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('iris_model.pkl')
iris = load_iris()

# App title
st.title("Iris Flower Prediction App")

st.write("Enter the flower's features to predict its species:")

# Sidebar input
sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
sepal_width = st.slider('Sepal width (cm)', 2.0, 4.4, 3.0)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 1.3)

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_species = iris.target_names[prediction]

# Display prediction
st.subheader("Prediction")
st.write(f"The predicted species is **{predicted_species}**.")

# Optional: Visualization
st.subheader("Feature Comparison")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    ax.scatter(df[df['species'] == i].iloc[:, 0], df[df['species'] == i].iloc[:, 2], label=iris.target_names[i], color=color)

ax.scatter(sepal_length, petal_length, color='black', label='Your Input', s=100)
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Petal length (cm)")
ax.legend()
st.pyplot(fig)
