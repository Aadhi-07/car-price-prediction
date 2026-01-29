### 📊 Car Price Prediction

Car Price Prediction is a machine learning project built in Python that predicts the selling price of a car based on its features. The project includes both model training and a prediction app using trained models and encoders. 
GitHub 

### 🧠 Project Overview

This project aims to take data about a used car (e.g., age, mileage, make/model, etc.) and use a trained machine learning model to estimate its selling price. It’s ideal for applications like resale value estimation tools or integrated features in automotive websites/apps.

### 📁 Repository Structure

Here’s what’s inside the project: 
```GitHub

├── app.py                     # Main prediction app (API/CLI)
├── train.py                   # Script to train and save the model
├── car_price_prediction.csv   # Dataset used for training
├── encoders.pkl               # Pre‑fit label encoders for categorical features
├── model.pkl                  # Trained machine learning model
├── requirements.txt           # Python dependencies
```
### 🛠️ Installation

Clone the repository
```
git clone https://github.com/Aadhi-07/car-price-prediction.git
cd car-price-prediction
```

Create a Python environment (optional but recommended)
```
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install dependencies
```
pip install -r requirements.txt
```

### 🧪 Train the Model

Run the training script to process the dataset, train the model, and save both the model and encoders:
```
python train.py
```

This will:

Read car_price_prediction.csv

Encode categorical variables

Train a machine learning regression model

Save the trained model (model.pkl)

Save fitted encoders (encoders.pkl)

### 🧩 Run the Prediction App

After training (or with the provided model.pkl and encoders.pkl), use the app to make predictions:
```
python app.py
```

You will be prompted (or you can modify it) to input features such as:

Year of manufacture

Mileage

Fuel type

Transmission type

Other relevant car features

The model will output the predicted selling price based on these inputs.

### 📦 Model & Encoders

model.pkl – Serialized machine learning regression model.

encoders.pkl – Serialized label encoders for preprocessing categorical features.

Both are loaded by the app to ensure consistent encoding and prediction at runtime.

###📊 Dataset

car_price_prediction.csv – Contains historical car sales data used to train the model.

Typically includes features like year, mileage, brand, etc.

Also includes the target variable (selling price).

### 🧠 How It Works (High‑Level)

Load dataset
Reads the CSV of car data.

Preprocess input
Encodes categorical features like fuel type or transmission using stored encoders.

Train model
Fits a regression model (e.g., linear regression, random forest, etc.) to predict car price.

Save model & encoders
Both are stored as .pkl files.

Predict
app.py loads model + encoders, preprocesses user input, and produces a price estimate.

### 📈 Technologies Used

Python

Scikit‑learn

Pandas / NumPy

Pickle for model serialization
