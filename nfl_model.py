'''
NFL MODEL

Standard Performance: Accuracy is in high 60s %
Desired Performance: 60%
Achieved Performance: 64%

Results from running Code:

Logistic Regression Results:
Accuracy: 0.6363636363636364
Mean Squared Error: 0.36363636363636365
Mean Absolute Error: 0.36363636363636365

Decision Tree Results:
Accuracy: 0.5
Mean Squared Error: 0.5
Mean Absolute Error: 0.5

Random Forest Results:
Accuracy: 0.6363636363636364
Mean Squared Error: 0.36363636363636365
Mean Absolute Error: 0.36363636363636365

Ensemble Model Results:
Accuracy: 0.6363636363636364
Mean Squared Error: 0.36363636363636365
Mean Absolute Error: 0.36363636363636365

The best model is: Logistic Regression with an accuracy of 0.6363636363636364

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# loading dataset
data = pd.read_csv('nfl_data.csv')

data.dropna(inplace=True)

# create result column
data['result'] = np.where(data['score_home'] > data['score_away'], 1, 0)

# encode categorical cols
categorical_columns = [
    'team_home', 'team_away', 'team_favorite_id', 'stadium', 'schedule_playoff', 'stadium_neutral'
]

label_encoders = {}

# add 'Encoded' to categorical cols
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column + 'Encoded'] = label_encoders[column].fit_transform(data[column])

# defining features to be trained on
x = data[[
    'schedule_season', 'schedule_playoffEncoded',
    'team_homeEncoded', 'team_awayEncoded', 'team_favorite_idEncoded',
    'spread_favorite', 'over_under_line', 'stadiumEncoded',
    'stadium_neutralEncoded', 'weather_temperature', 'weather_wind_mph', 'weather_humidity'
]]

# target variable
y = data['result']

# split data into trainig and testing sets 
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=92005)

# models to be tested
log_reg = LogisticRegression(max_iter=1000, random_state=92003)
dt = DecisionTreeClassifier(random_state=92003)
rf = RandomForestClassifier(random_state=92003, n_jobs=-1)

# train and evaluate individual models
models = {
    'Logistic Regression': log_reg,
    'Decision Tree': dt,
    'Random Forest': rf
}

# keeping track of best model
best_model_name = None
best_model = None
best_accuracy = 0

# train and evaluate each model
for model_name, model in models.items():
    model.fit(x_Train, y_Train)
    y_Pred = model.predict(x_Test)
    
    accuracy = accuracy_score(y_Test, y_Pred)
    mse = mean_squared_error(y_Test, y_Pred)
    mae = mean_absolute_error(y_Test, y_Pred)
    
    print(f'{model_name} Results:')
    print(f'Accuracy: {accuracy}')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}\n')
    
    # update best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

# ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('log_reg', log_reg), 
    ('dt', dt), 
    ('rf', rf)
], voting='hard')

# train ensemble
ensemble_model.fit(x_Train, y_Train)
y_Pred_ensemble = ensemble_model.predict(x_Test)

# evaluating ensemble model
accuracy_ensemble = accuracy_score(y_Test, y_Pred_ensemble)
mse_ensemble = mean_squared_error(y_Test, y_Pred_ensemble)
mae_ensemble = mean_absolute_error(y_Test, y_Pred_ensemble)

print('Ensemble Model Results:')
print(f'Accuracy: {accuracy_ensemble}')
print(f'Mean Squared Error: {mse_ensemble}')
print(f'Mean Absolute Error: {mae_ensemble}\n')

# compare ensemble model with best individual model
if accuracy_ensemble > best_accuracy:
    best_model_name = 'Ensemble Model'
    best_model = ensemble_model
    best_accuracy = accuracy_ensemble

print(f'The best model is: {best_model_name} with an accuracy of {best_accuracy}')