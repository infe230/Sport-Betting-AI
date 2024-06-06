from flask import Flask, request, jsonify, render_template

from soccer_model import predict_winner_soccer, x, label_encoders
# from nfl_model import predict_winner_nfl, x, label_encoders
# from nba_model import predict_winner_nba, x, label_encoders
# from ufc_model import predict_winner_ufc, x, label_encoders


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    home_side = request.form['home_side']
    away_side = request.form['away_side']
    model_type = request.form['model_type']

    # Dummy prediction logic for example purposes
    predictions = {}
    if model_type == 'premier-league':
        predictions = predict_winner_soccer(home_side, away_side, x, label_encoders)

    # elif model_type == 'nba':
        # predictions = predict_winner_nba(home_side, away_side, x, label_encoders)

    # elif model_type == 'nfl':
     #    predictions = predict_winner_nfl(home_side, away_side, x, label_encoders)

    # elif model_type == 'ufc':
        # predictions = predict_winner_ufc(home_side, away_side, x, label_encoders)


    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
