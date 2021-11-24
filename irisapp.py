from flask import Flask,render_template,request
import pickle
import numpy as np

model= pickle.load(open('iris_model.pkl','rb'))


irisapp= Flask(__name__)


@irisapp.route('/')
def man():
    return render_template('home.html')

@irisapp.route('/predict',methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr= np.array([[data1, data2,data3,data4]])
    pred= model.predict(arr)
    return render_template('after.html',data=pred)

@irisapp.route('/test')
def test2():
    return 'this is not the home page'

if __name__== "__main__":
    irisapp.run(debug=True)