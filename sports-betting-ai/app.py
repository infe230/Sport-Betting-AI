from flask import Flask, request, jsonify, render_template

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
        predictions = {
            'home_win_prob': 0.7,
            'away_win_prob': 0.3
        }
    elif model_type == 'nba':
        predictions = {
            'home_win_prob': 0.6,
            'away_win_prob': 0.4
        }
    elif model_type == 'tennis':
        predictions = {
            'home_win_prob': 0.8,
            'away_win_prob': 0.2
        }
    elif model_type == 'ufc':
        predictions = {
            'home_win_prob': 0.5,
            'away_win_prob': 0.5
        }

    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
