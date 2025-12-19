import streamlit as st
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MODEL_FILE = "knn_model_file.pkl"

st.title("KNN Model Example")

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Check if model file exists
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded")
else:
    # STEP 5: Train model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # STEP 6: Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    st.success("✅ Model trained and saved")

# Input
a = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
b = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
c = st.number_input("Petal Length", 0.0, 10.0, 1.4)
d = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    result = model.predict([[a, b, c, d]])
    st.write("Prediction:", result[0])
