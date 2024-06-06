'''
UFC MODEL

Standard Performance: Stochastic Gradient Descent, Multilayer Perceptron and Decision Tree - Accuracy is just over 60%
Desired Performance: 60%
Achieved Performance: 71%

Results from running Code:

Linear Regression Results:
Accuracy: 0.7132757266300078
Mean Squared Error: 0.19372117902082422
Mean Absolute Error: 0.3951156336908637

Random Forest Results:
Accuracy: 0.6967792615868028
Mean Squared Error: 0.3032207384131972
Mean Absolute Error: 0.3032207384131972

Logistic Regression Results:
Accuracy: 0.7124901806755696
Mean Squared Error: 0.2875098193244305
Mean Absolute Error: 0.2875098193244305

Decision Tree Results:
Accuracy: 0.608012568735271
Mean Squared Error: 0.391987431264729
Mean Absolute Error: 0.391987431264729

Neural Network Results:
Accuracy: 0.6417910447761194
Mean Squared Error: 0.3582089552238806
Mean Absolute Error: 0.3582089552238806

The best model is: Linear Regression with an accuracy of 0.7132757266300078

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
data = pd.read_csv('Datasets/ufc_dataset.csv')

# drop missing vals
data.dropna(inplace=True)

# encode categorical cols
categorical_columns = [
    'event_name', 'r_fighter', 'b_fighter', 'winner', 'weight_class', 'is_title_bout', 'gender', 'method', 'referee', 'r_stance', 'b_stance'
]

label_encoders = {}

# add 'Encoded' to categorical cols
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column + 'Encoded'] = label_encoders[column].fit_transform(data[column])

# defining features to be trained on
x = data[
    [
        'event_nameEncoded', 'r_fighterEncoded', 'b_fighterEncoded', 'weight_classEncoded', 
        'is_title_boutEncoded', 'genderEncoded', 'r_wins_total', 'r_losses_total', 'r_age', 
        'r_height', 'r_weight', 'r_reach', 'r_stanceEncoded', 'r_SLpM_total', 'r_SApM_total', 
        'r_sig_str_acc_total', 'r_td_acc_total', 'r_str_def_total', 'r_td_def_total', 
        'r_sub_avg', 'r_td_avg', 'b_wins_total', 'b_losses_total', 'b_age', 'b_height', 
        'b_weight', 'b_reach', 'b_stanceEncoded', 'b_SLpM_total', 'b_SApM_total', 
        'b_sig_str_acc_total', 'b_td_acc_total', 'b_str_def_total', 'b_td_def_total', 
        'b_sub_avg', 'b_td_avg'
    ]
]

# target variable
y = data['winnerEncoded']

# split data into trainig and testing sets 
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=92003)

# models to be tested
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(random_state=92003),
    'Logistic Regression': LogisticRegression(max_iter=4000),
    'Decision Tree': DecisionTreeClassifier(random_state=92003),
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
    
    # update the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f'The best model is: {best_model_name} with an accuracy of {best_accuracy}')
