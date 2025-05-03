import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset 
data = pd.read_csv("tested.csv")

# Feature Engineering

data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                      'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Family features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = np.where(data['FamilySize'] > 1, 0, 1)

# Cabin features
data['Deck'] = data['Cabin'].str[0]
data['Deck'] = data['Deck'].fillna('Unknown')

# Age binning
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Handle Missing Values
# ---------------------
# Age - median imputation
data['Age'] = data['Age'].fillna(data['Age'].median())

# Embarked - mode imputation
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Fare - median imputation (just in case)
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Categorical Encoding
# ---------------------
# One-hot encoding
cat_features = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
data = pd.get_dummies(data, columns=cat_features, drop_first=True)

# Drop unnecessary columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'], axis=1)

# Split Data
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (optional for tree-based models)
scaler = StandardScaler()
num_features = ['Fare', 'FamilySize']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print("Preprocessed Training Shape:", X_train.shape)
print("Available Features:", list(X_train.columns))