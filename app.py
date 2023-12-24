

import pickle

import numpy as np
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

model=pickle.load(open('units_model.pkl','rb'))

model1=pickle.load(open('load_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    prediction1=model1.predict(final)
    load='{}'.format(prediction1[0])
    unit='{}'.format(prediction[0])
    return render_template('index.html',loads=load,units=unit)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=145)





