from flask import Flask,render_template,request
import pickle
import numpy as np
from joblib import load
# Load the model using joblib

# Use the loaded model for predictions

model = load('model.pkl')
print(model)
app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_phishing():

    features = [
        float(request.form.get('url_length')),
        float(request.form.get('n_dots')),
        float(request.form.get('n_hypens')),
        float(request.form.get('n_underline')),
        float(request.form.get('n_slash')),
        float(request.form.get('n_questionmark')),
        float(request.form.get('n_equal')),
        float(request.form.get('n_at')),
        float(request.form.get('n_and')),
        float(request.form.get('n_exclamation')),
        float(request.form.get('n_space')),
        float(request.form.get('n_tilde')),
        float(request.form.get('n_comma')),
        float(request.form.get('n_plus')),
        float(request.form.get('n_asterisk')),
        float(request.form.get('n_hastag')),
        float(request.form.get('n_dollar')),
        float(request.form.get('n_percent')),
        float(request.form.get('n_redirection'))
    ]

    # Make prediction
    result = model.predict([features])


    # url_length = float(request.form.get('url_length'))
    # n_dots = float(request.form.get('n_dots'))
    # n_hypens = float(request.form.get('n_hypens'))
    # n_underline = float(request.form.get('n_underline'))
    # n_slash = float(request.form.get('n_slash'))
    # n_questionmark = float(request.form.get('n_questionmark'))
    # n_equal = float(request.form.get('n_equal'))
    # n_at = float(request.form.get('n_at'))
    # n_and = float(request.form.get('n_and'))
    # n_exclamation = float(request.form.get('n_exclamation'))
    # n_space = float(request.form.get('n_space'))
    # n_tilde = float(request.form.get('n_tilde'))
    # n_comma = float(request.form.get('n_comma'))
    # n_plus = float(request.form.get('n_plus'))
    # n_asterisk = float(request.form.get('n_asterisk'))
    # n_hastag = float(request.form.get('n_hastag'))
    # n_dollar = float(request.form.get('n_dollar'))
    # n_percent = float(request.form.get('n_percent'))
    # n_redirection = float(request.form.get('n_redirection'))


    # #prediction
    # result = model.predict(np.array([
    # url_length,
    # n_dots,
    # n_hypens,
    # n_underline,
    # n_slash,
    # n_questionmark,
    # n_equal,
    # n_at,
    # n_and,
    # n_exclamation,
    # n_space,
    # n_tilde,
    # n_comma,
    # n_plus,
    # n_asterisk,
    # n_hastag,
    # n_dollar,
    # n_percent,
    # n_redirection

    # ]))
    if result[0] == 1:
        result = 'Legitimate'
    else:
        result = 'Phishing'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(debug=True)