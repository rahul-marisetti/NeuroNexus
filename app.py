from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained model and scaler
model = joblib.load('titanic_rf_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    passenger_data = {
        'Pclass': int(request.form['pclass']),
        'Name': request.form['name'],
        'Sex': request.form['sex'],
        'Age': float(request.form['age']),
        'SibSp': int(request.form['sibsp']),
        'Parch': int(request.form['parch']),
        'Fare': float(request.form['fare']),
        'Cabin': request.form['cabin'],
        'Embarked': request.form['embarked']
    }
    
    # Preprocessing
    df = pd.DataFrame([passenger_data])
    
    # Feature Engineering
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] > 1, 0, 1)
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # One-hot encoding
    cat_features = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)
    
    # Align columns with training data
    train_columns = model.feature_names_in_
    df = df.reindex(columns=train_columns, fill_value=0)
    
    # Scale numerical features
    num_features = ['Fare', 'FamilySize']
    df[num_features] = scaler.transform(df[num_features])
    
    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1][0]
    
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    return render_template('result.html', result=result, probability=probability*100)

if __name__ == '__main__':
    app.run(debug=True)