'''
NFL MODEL

Standard Performance: Low to Mid 70s %
Desired Performance: 65%
Achieved Performance: 61%

Results from running Code:

Linear Regression Results:
Accuracy: 0.6076169209775462
Mean Squared Error: 0.23789853340272424
Mean Absolute Error: 0.477419743889817

Random Forest Results:
Accuracy: 0.6123309763056692
Mean Squared Error: 0.3876690236943307
Mean Absolute Error: 0.3876690236943307

Logistic Regression Results:
Accuracy: 0.6076169209775462
Mean Squared Error: 0.3923830790224538
Mean Absolute Error: 0.3923830790224538

Decision Tree Results:
Accuracy: 0.5772236695199107
Mean Squared Error: 0.42277633048008934
Mean Absolute Error: 0.42277633048008934

The best model is: Random Forest with an accuracy of 0.6123309763056692

'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# loading dataset
data = pd.read_csv('3ptToWin.csv')

# encode categorical cols
categorical_columns = [
    'MATCHUP', 'TEAM_NAME_HOME', 'TEAM_ABBREVIATION_HOME', 'TEAM_NAME_AWAY', 'TEAM_ABBREVIATION_AWAY', 'WL_HOME'
]

label_encoders = {}

# add 'Encoded' to categorical cols
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column + 'Encoded'] = label_encoders[column].fit_transform(data[column])

# defining features to be trained on
x = data[[
    'GAME_ID', 'MATCHUPEncoded', 'TEAM_NAME_HOMEEncoded', 'TEAM_ID_HOME',
    'TEAM_NAME_AWAYEncoded', 'TEAM_ID_AWAY'
]]

# target variable
y = data['WL_HOMEEncoded']

# split data into trainig and testing sets 
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=92001)

# models to be tested
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(random_state=92001),
    'Logistic Regression': LogisticRegression(max_iter=4000),
    'Decision Tree': DecisionTreeClassifier(random_state=92001)
}

# keeping track of best model
best_model_name = None
best_model = None
best_accuracy = 0

# train and evaluate each model
for model_name, model in models.items():
    # train the model
    model.fit(x_Train, y_Train)
    
    # making predictions
    y_Prediction = model.predict(x_Test)
    
    # Evaludate performance for each model
    if model_name == 'Linear Regression':
        y_Prediction = np.round(y_Prediction).astype(int)
    accuracy = accuracy_score(y_Test, y_Prediction)
    mse = mean_squared_error(y_Test, y_Prediction)
    mae = mean_absolute_error(y_Test, y_Prediction)
    
    print(f'{model_name} Results:')
    print(f'Accuracy: {accuracy}')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}\n')
    
    # update  best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f'The best model is: {best_model_name} with an accuracy of {best_accuracy}')