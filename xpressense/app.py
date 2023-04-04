import logging

from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('modelAcousticExp.pkl', 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return 'Equipped with his five senses, man explores the universe around him and calls the adventure Science. –- Edwin Hubble'


@app.route('/predict', methods=['POST'])
def predict():
        amp = request.form.get('amp',type=float)
        phase = request.form.get('phase',type=float)
        # amp=0.00;
        # phase = 0.00;
        expression = {'amp': amp, 'phase': phase}
        input_query = np.array([[amp, phase]])
        expression = model.predict(input_query)[0]
        return jsonify({'expression:': expression})



if __name__ == '__main__':
    app.run(debug=True)
