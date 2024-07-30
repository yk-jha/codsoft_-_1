#required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report

#data loading
df = pd.read_csv('Titanic-Dataset.csv')
#data preprocessing
df = df.dropna(subset=['Survived'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['PassengerId' , 'Age' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' , 'Fare' , 'Cabin'])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
features = df.drop(columns=['Survived'])
target = df['Survived']

#splitting Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Train and fit Model
model = LogisticRegression()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#Evaluation

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
