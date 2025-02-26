# Titanic Survival Prediction

This project builds a predictive model to determine whether a passenger on the Titanic would have survived based on various features such as age, ticket class, fare, gender, and port of embarkation. The model is trained using a neural network and can be used to make predictions on new data.

## Project Structure

- `Titanic-Dataset.csv`: The dataset used for training the model.
- `Titanic_Survival_Model.ipynb`: Code for data preprocessing, feature selection, and training the neural network.
- `Titanic_Prediction.py`: A script that loads the trained model and takes user input to predict survival.
- `titanic_survival_model.h5`: The saved trained model for future use.
- `README.md`: This file, providing an overview of the project.

## Requirements

To run this project, install the required dependencies using:

```sh
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow keras
```

## Model Training

The `Titanic_Survival_Model.py` script performs the following steps:

1. Loads the dataset and performs exploratory data analysis.
2. Handles missing values and encodes categorical features.
3. Selects the top 5 features correlated with survival.
4. Preprocesses the data and trains a deep learning model with a mix of layers.
5. Saves the trained model as `titanic_survival_model.h5`.

## Prediction Script

The `Titanic_Prediction.py` script allows users to enter their details, preprocesses the input data, and predicts survival using the trained model.

## Usage

Run the prediction script using:

```sh
python Titanic_Prediction.py
```

Follow the prompts to enter passenger details, and the model will predict survival.

##

