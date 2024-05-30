'''
PREMIER LEAGUE (SOCCER) MODEL

Standard Performance:  Logistic Regression + Decision Tree = 70% test accuracy
Desired Performance: 65%
Achieved Performance: 56%

Results from running Code:

Random Forest Results:
Accuracy: 0.5522788203753352
Mean Squared Error: 1.0107238605898123
Mean Absolute Error: 0.6353887399463807

Linear Regression Results:
Accuracy: 0.3646112600536193
Mean Squared Error: 0.7319034852546917
Mean Absolute Error: 0.6675603217158177

Logistic Regression Results:
Accuracy: 0.5576407506702413
Mean Squared Error: 1.0536193029490617
Mean Absolute Error: 0.646112600536193

Decision Tree Results:
Accuracy: 0.4691689008042895
Mean Squared Error: 1.1742627345844503
Mean Absolute Error: 0.7453083109919572

The best model is: Logistic Regression with an accuracy of 0.5576407506702413
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# loading dataset
data = pd.read_csv('PremierLeague.csv')

# drop missing vals
data.dropna(inplace=True)

# encode categorical cols
categorical_columns = ['HomeTeam', 'AwayTeam', 'FullTimeResult', 'HalfTimeResult', 'Referee']
label_encoders = {}

# add 'Encoded' to categorical cols
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column + 'Encoded'] = label_encoders[column].fit_transform(data[column])

# calc avg stats
average_stats = data.groupby('HomeTeam').agg({
    'FullTimeHomeTeamGoals': 'mean',
    'FullTimeAwayTeamGoals': 'mean',
    'HalfTimeHomeTeamGoals': 'mean',
    'HalfTimeAwayTeamGoals': 'mean',
    'HomeTeamShots': 'mean',
    'AwayTeamShots': 'mean',
    'HomeTeamShotsOnTarget': 'mean',
    'AwayTeamShotsOnTarget': 'mean',
    'HomeTeamCorners': 'mean',
    'AwayTeamCorners': 'mean',
    'HomeTeamFouls': 'mean',
    'AwayTeamFouls': 'mean',
    'HomeTeamYellowCards': 'mean',
    'AwayTeamYellowCards': 'mean',
    'HomeTeamRedCards': 'mean',
    'AwayTeamRedCards': 'mean'
}).reset_index()

average_stats.columns = [
    'Team', 'AvgFullTimeHomeTeamGoals', 'AvgFullTimeAwayTeamGoals', 'AvgHalfTimeHomeTeamGoals',
    'AvgHalfTimeAwayTeamGoals', 'AvgHomeTeamShots', 'AvgAwayTeamShots', 'AvgHomeTeamShotsOnTarget',
    'AvgAwayTeamShotsOnTarget', 'AvgHomeTeamCorners', 'AvgAwayTeamCorners', 'AvgHomeTeamFouls',
    'AvgAwayTeamFouls', 'AvgHomeTeamYellowCards', 'AvgAwayTeamYellowCards', 'AvgHomeTeamRedCards',
    'AvgAwayTeamRedCards'
]

# merging avg stats
data = data.merge(average_stats, how='left', left_on='HomeTeam', right_on='Team').drop('Team', axis=1)
data = data.merge(average_stats, how='left', left_on='AwayTeam', right_on='Team', suffixes=('_Home', '_Away')).drop('Team', axis=1)

# defining features to be trained on
x = data[[
    'HomeTeamEncoded', 'AwayTeamEncoded', 'RefereeEncoded', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam',
    'B365Over2.5Goals', 'B365Under2.5Goals', 'MarketMaxHomeTeam', 'MarketMaxDraw', 'MarketMaxAwayTeam',
    'MarketAvgHomeTeam', 'MarketAvgDraw', 'MarketAvgAwayTeam', 'MarketMaxOver2.5Goals', 'MarketMaxUnder2.5Goals',
    'MarketAvgOver2.5Goals', 'MarketAvgUnder2.5Goals', 'AvgFullTimeHomeTeamGoals_Home', 'AvgFullTimeAwayTeamGoals_Home',
    'AvgHalfTimeHomeTeamGoals_Home', 'AvgHalfTimeAwayTeamGoals_Home', 'AvgHomeTeamShots_Home', 'AvgAwayTeamShots_Home',
    'AvgHomeTeamShotsOnTarget_Home', 'AvgAwayTeamShotsOnTarget_Home', 'AvgHomeTeamCorners_Home', 'AvgAwayTeamCorners_Home',
    'AvgHomeTeamFouls_Home', 'AvgAwayTeamFouls_Home', 'AvgHomeTeamYellowCards_Home', 'AvgAwayTeamYellowCards_Home',
    'AvgHomeTeamRedCards_Home', 'AvgAwayTeamRedCards_Home', 'AvgFullTimeHomeTeamGoals_Away', 'AvgFullTimeAwayTeamGoals_Away',
    'AvgHalfTimeHomeTeamGoals_Away', 'AvgHalfTimeAwayTeamGoals_Away', 'AvgHomeTeamShots_Away', 'AvgAwayTeamShots_Away',
    'AvgHomeTeamShotsOnTarget_Away', 'AvgAwayTeamShotsOnTarget_Away', 'AvgHomeTeamCorners_Away', 'AvgAwayTeamCorners_Away',
    'AvgHomeTeamFouls_Away', 'AvgAwayTeamFouls_Away', 'AvgHomeTeamYellowCards_Away', 'AvgAwayTeamYellowCards_Away',
    'AvgHomeTeamRedCards_Away', 'AvgAwayTeamRedCards_Away'
]]

# target variable
y = data['FullTimeResultEncoded']

# split data into trainig and testing sets 
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=92002)

# models to be tested
models = {
    'Random Forest': RandomForestClassifier(random_state=92002),
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(max_iter=3000),
    'Decision Tree': DecisionTreeClassifier(random_state=92002)
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
    
    # update best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f'The best model is: {best_model_name} with an accuracy of {best_accuracy}')