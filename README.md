# Student Depression Prediction Streamlit App

This app uses the **Student Depression Dataset** from **Kaggle** to build a prediction model. It also allows saving user input data into a local **user_data.csv** file. The model is trained using **Random Forest**, features are scaled via **StandardScaler**, and the interface is powered by **Streamlit**.

## Project structure

.DEPRESSION_PREDICTING

├── student_depression_predictor.py # Main Streamlit application

├── depression_model.pkl # Trained RandomForest model file

├── scaler.pkl #StandardScaler object

└── requirements.txt # Required libraries

## Requirements

You need the following Python libraries:

- streamlit  
- scikit-learn  
- pandas  
- numpy  
- joblib  

Install via:

```bash
pip install streamlit scikit-learn pandas numpy joblib

```
## How to run

Clone or download this repository

Make sure model.pkl and scaler.pkl are present

Ensure user_data.csv exists (it can be initially empty or with header row)

**Run the app**:

```
streamlit run student_depression_predictor.py

```
The app opens in your browser, typically at http://localhost:8501

## Workflow & Behavior
 
The base dataset is from Kaggle and used to train a Random Forest classification model.

Feature scaling is done using StandardScaler.

In the Streamlit app, users enter input values; the app scales them, makes a prediction, and displays the result.

The app also appends the user’s input values (and optionally the prediction) into user_data.csv for storage and future reference. 
