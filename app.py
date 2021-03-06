import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results in HTML
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = np.round(prediction, 0)
    output = int(output)
    output = "${:,.2f}".format(output)
    
    return render_template('index.html', prediction_text='House price should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
