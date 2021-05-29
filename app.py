from flask import Flask, render_template, url_for, request
import numpy as np
import pickle


app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    var = request.form['Text']
    prediction=model.predict([var])
    output=format(prediction[0])
    return render_template('home.html', pred='The news is: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)