from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, metrics ensemble

filename= 'botsinthenet.pkl'
LogReg= pickle.load(open(filename, 'rb'))
vectorizer= pickle.load(open('transform.pkl', 'rb'))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('web.html')

@app.route('/predict', methods= ['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data= [message]
        log = vectorizer.transform(data).toarray()
        pred = LogReg(log)
    return render_template('web.html', prediction=pred)


if __name__=='__main__':
    app.run(debug=True)
