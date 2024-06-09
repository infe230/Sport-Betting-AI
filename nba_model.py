'''
NBA MODEL

Standard Performance: Low to Mid 70s %
Desired Performance: 65%
Achieved Performance: 62%

Results from running Code:

Linear Regression Results:
Accuracy: 0.6076169209775462
Mean Squared Error: 0.23789853340272424
Mean Absolute Error: 0.477419743889817

Random Forest Results:
Accuracy: 0.6123309763056692
Mean Squared Error: 0.3876690236943307
Mean Absolute Error: 0.3876690236943307
Accuracy with Tuning: 0.6205185460860936
Change: +0.01

Logistic Regression Results:
Accuracy: 0.6076169209775462
Mean Squared Error: 0.3923830790224538
Mean Absolute Error: 0.3923830790224538

Decision Tree Results:
Accuracy: 0.5772236695199107
Mean Squared Error: 0.42277633048008934
Mean Absolute Error: 0.42277633048008934
Accuracy with Tuning: 0.6136955712690734
Change: +0.04

Neural Network Results:
Accuracy: 0.6076169209775462
Mean Squared Error: 0.3923830790224538
Mean Absolute Error: 0.3923830790224538

The best model is: Random Forest with an accuracy of 0.6205185460860936

'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier


# loading dataset
data = pd.read_csv('Datasets/nba_dataset.csv')

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
    'Random Forest': RandomForestClassifier(random_state=92001, max_depth = 9, min_samples_split = 2, min_samples_leaf = 9),
    'Logistic Regression': LogisticRegression(max_iter=4000),
    'Decision Tree': DecisionTreeClassifier(random_state=92001, max_depth = 6, min_samples_split = 2, min_samples_leaf = 2),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000, random_state=92003)

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

def predict_winner_nba(home_team, away_team, data, label_encoders):
    home_team_encoded = label_encoders['TEAM_NAME_HOME'].transform([home_team])[0]
    away_team_encoded = label_encoders['TEAM_NAME_AWAY'].transform([away_team])[0]
    
    home_team_data = data[data['TEAM_NAME_HOMEEncoded'] == home_team_encoded].iloc[0]
    away_team_data = data[data['TEAM_NAME_AWAYEncoded'] == away_team_encoded].iloc[0]
    
    input_data = np.array([
        home_team_data['GAME_ID'], home_team_data['MATCHUPEncoded'], home_team_encoded, home_team_data['TEAM_ID_HOME'],
        away_team_encoded, away_team_data['TEAM_ID_AWAY']
    ]).reshape(1, -1)
    
    probabilities = best_model.predict_proba(input_data)[0]
    
    result = {
        'home_win_prob': probabilities[0],
        'draw_prob': 0,
        'away_win_prob': probabilities[1]
    }
    
    return result

