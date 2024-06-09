'''
PREMIER LEAGUE (SOCCER) MODEL

Standard Performance:  Logistic Regression + Decision Tree = 70% test accuracy
Desired Performance: 65%
Achieved Performance: 57%

Results from running Code:

Random Forest Results:
Accuracy: 0.5522788203753352
Mean Squared Error: 1.0107238605898123
Mean Absolute Error: 0.6353887399463807
Accuracy with Tuning: 0.5630026809651475
Change: +0.01

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
Accuracy with Tuning: 0.5603217158176944
Change: +0.09

Neural Network Results:
Accuracy: 0.5710455764075067
Mean Squared Error: 1.0241286863270778
Mean Absolute Error: 0.6273458445040214

The best model is: Neural Network with an accuracy of 0.5710455764075067
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier


# loading dataset
data = pd.read_csv('Datasets/PremierLeague.csv')

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
    'Random Forest': RandomForestClassifier(random_state=92002, n_estimators=59, max_depth=3, min_samples_split=2, min_samples_leaf=9),
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(max_iter=3000),
    'Decision Tree': DecisionTreeClassifier(random_state=92002, max_depth= 3, min_samples_split=2, min_samples_leaf = 17),
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
    
    # update best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f'The best model is: {best_model_name} with an accuracy of {best_accuracy}')


def predict_winner_soccer(home_team, away_team, data, label_encoders):
    # Check if 'HomeTeam' and 'AwayTeam' encoders exist
    if 'HomeTeam' not in label_encoders or 'AwayTeam' not in label_encoders:
        raise KeyError("The label encoders for 'HomeTeam' or 'AwayTeam' are not found.")

    # Encode team names
    home_team_encoded = label_encoders['HomeTeam'].transform([home_team])[0]
    away_team_encoded = label_encoders['AwayTeam'].transform([away_team])[0]

    # Fetching average stats for both teams
    home_stats = data[data['HomeTeamEncoded'] == home_team_encoded].iloc[0]
    away_stats = data[data['AwayTeamEncoded'] == away_team_encoded].iloc[0]

    # Creating input for prediction
    input_data = np.array([
        home_team_encoded, away_team_encoded, home_stats['RefereeEncoded'],
        home_stats['B365HomeTeam'], home_stats['B365Draw'], home_stats['B365AwayTeam'],
        home_stats['B365Over2.5Goals'], home_stats['B365Under2.5Goals'], home_stats['MarketMaxHomeTeam'],
        home_stats['MarketMaxDraw'], home_stats['MarketMaxAwayTeam'], home_stats['MarketAvgHomeTeam'],
        home_stats['MarketAvgDraw'], home_stats['MarketAvgAwayTeam'], home_stats['MarketMaxOver2.5Goals'],
        home_stats['MarketMaxUnder2.5Goals'], home_stats['MarketAvgOver2.5Goals'], home_stats['MarketAvgUnder2.5Goals'],
        home_stats['AvgFullTimeHomeTeamGoals_Home'], home_stats['AvgFullTimeAwayTeamGoals_Home'],
        home_stats['AvgHalfTimeHomeTeamGoals_Home'], home_stats['AvgHalfTimeAwayTeamGoals_Home'],
        home_stats['AvgHomeTeamShots_Home'], home_stats['AvgAwayTeamShots_Home'],
        home_stats['AvgHomeTeamShotsOnTarget_Home'], home_stats['AvgAwayTeamShotsOnTarget_Home'],
        home_stats['AvgHomeTeamCorners_Home'], home_stats['AvgAwayTeamCorners_Home'],
        home_stats['AvgHomeTeamFouls_Home'], home_stats['AvgAwayTeamFouls_Home'],
        home_stats['AvgHomeTeamYellowCards_Home'], home_stats['AvgAwayTeamYellowCards_Home'],
        home_stats['AvgHomeTeamRedCards_Home'], home_stats['AvgAwayTeamRedCards_Home'],
        away_stats['AvgFullTimeHomeTeamGoals_Away'], away_stats['AvgFullTimeAwayTeamGoals_Away'],
        away_stats['AvgHalfTimeHomeTeamGoals_Away'], away_stats['AvgHalfTimeAwayTeamGoals_Away'],
        away_stats['AvgHomeTeamShots_Away'], away_stats['AvgAwayTeamShots_Away'],
        away_stats['AvgHomeTeamShotsOnTarget_Away'], away_stats['AvgAwayTeamShotsOnTarget_Away'],
        away_stats['AvgHomeTeamCorners_Away'], away_stats['AvgAwayTeamCorners_Away'],
        away_stats['AvgHomeTeamFouls_Away'], away_stats['AvgAwayTeamFouls_Away'],
        away_stats['AvgHomeTeamYellowCards_Away'], away_stats['AvgAwayTeamYellowCards_Away'],
        away_stats['AvgHomeTeamRedCards_Away'], away_stats['AvgAwayTeamRedCards_Away']
    ]).reshape(1, -1)

    # Convert input data to DataFrame with the same column names as the training data
    feature_names = [
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
    ]
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Predict probabilities
    probabilities = best_model.predict_proba(input_df)[0]
    result = {
        'home_win_prob': probabilities[0],
        'draw_prob': probabilities[1],
        'away_win_prob': probabilities[2]
    }
    
    return result
