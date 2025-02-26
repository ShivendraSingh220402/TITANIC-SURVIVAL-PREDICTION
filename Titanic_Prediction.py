import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("titanic_survival_model.h5")
top_features = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Age']

print("Enter Passenger Details for Survival Prediction")

pclass = int(input("Ticket Class (1 = First, 2 = Second, 3 = Third): "))
sex = input("Sex (male/female): ").strip().lower()
fare = float(input("Fare Amount (in currency value): "))
embarked = input("Embarked Port (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()
age = float(input("Age: "))

sex = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_mapping.get(embarked, 2)

user_data = np.array([[pclass, sex, fare, embarked, age]])
scaler = StandardScaler()
user_data = scaler.fit_transform(user_data)
user_data = np.expand_dims(user_data, axis=-1)

# Predict survival
prediction = model.predict(user_data)
survival = "Survived" if prediction[0][0] > 0.5 else "Did Not Survive"

print(f"Prediction: {survival}")
