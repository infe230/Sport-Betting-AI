from flask import Flask, request, jsonify, render_template

from soccer_model import predict_winner_soccer, x as soccer_x, label_encoders as soccer_encoders
from nba_model import predict_winner_nba, x as nba_x, label_encoders as nba_encoders
from nfl_model import predict_winner_nfl, x as nfl_x, label_encoders as nfl_encoders
from ufc_model import predict_winner_ufc, x as ufc_x, label_encoders as ufc_encoders


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    home_side = request.form['home_side']
    away_side = request.form['away_side']
    model_type = request.form['model_type']

    predictions = {}
    if model_type == 'premier-league':
        predictions = predict_winner_soccer(home_side, away_side, soccer_x, soccer_encoders)

    elif model_type == 'nba':
        predictions = predict_winner_nba(home_side, away_side, nba_x, nba_encoders)

    elif model_type == 'nfl':
        predictions = predict_winner_nfl(home_side, away_side, nfl_x, nfl_encoders)

    elif model_type == 'ufc':
        predictions = predict_winner_ufc(home_side, away_side, ufc_x, ufc_encoders)


    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
