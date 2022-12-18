from flask import Flask,render_template, request
import numpy as np            
import pickle
import pandas as pd    

model = pickle.load(open('lasso_reg_model.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def my():
    return render_template("home.html" )

@app.route("/predict",methods = ["POST", "GET"])
def home():
    a = eval(request.form['bedrooms'])
    b = eval(request.form['bathrooms'])
    c = eval(request.form['stories'])
    d = eval(request.form['parking'])
    e = eval(request.form['furnishingstatus'])
    f = eval(request.form['area'])
    



    arr = np.array([[a, b, c, d, e, f]])
    pred = model.predict(arr)
    return render_template('after.html', prediction_text = pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)   
   

# @app.route('/predict1')
# def predict1():
#     a = eval(request.args.get('a'))
#     b = eval(request.args.get('b'))
#     c = eval(request.args.get('c'))
#     d = eval(request.args.get('d'))

#     arr = np.array([[a, b, c, d, e, f, g, h, i, j]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

    