from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
@app.route('/')
def MINI_PRO():
    return render_template('MINI_PRO.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route('/heart_dis')
def heart_dis():
    return render_template('heart_dis.html')
@app.route("/kidney")
def kidney():
    return render_template('kidney.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetes/', methods=['POST'])
def predict1():
    if request.method == 'POST':
        # Retrieve form data using the names assigned to the input fields
        pregnancies = request.form.get('pregnancies')
        glucose = float(request.form.get('glucose'))
        blood_pressure = float(request.form.get('blood_pressure'))
        skin_thikness = float(request.form.get('skin_thikness'))
        insulin = float(request.form.get("insulin"))
        bmi = float(request.form.get("bmi"))
        dpf = float(request.form.get("dpf"))
        age = request.form.get('age')

        # Load your trained model
        with open('mini_project.pkl', 'rb') as file:
            model = pickle.load(file)

        # Perform prediction using your model
        input_data = [[pregnancies, glucose, blood_pressure,skin_thikness,insulin,bmi,dpf,age]]
        # Create a list of input features
        print("Input Data:", input_data)


        input_data_numeric = np.array(input_data).astype(float)

        prediction = model.predict(input_data_numeric)

        prediction = model.predict(input_data)
        print(prediction)

        # Render the prediction result on a template
        return render_template('output.html', prediction=prediction)

@app.route('/heart_dis/', methods=['POST'])
def predict2():
    if request.method == 'POST':
        # Retrieve form data using the names assigned to the input fields
        age = float(request.form.get('age'))
        gender = float(request.form.get('gender'))
        cp = float(request.form.get('cp'))
        trest = float(request.form.get('trest'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        rest = float(request.form.get('rest'))
        thalach = float(request.form.get('thal'))
        exang = float(request.form.get('exang'))
        old = float(request.form.get('old'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))

        # Load your trained model
        with open('mini_project2.pkl', 'rb') as file:
            model2 = pickle.load(file)

        # Perform prediction using your model
        input_data = [[age, gender, cp, trest, chol, fbs, rest, thalach, exang, old, slope, ca, thal]]
        input_array = np.array(input_data).astype(float)
        prediction = model2.predict(input_array)

        # Render the prediction result on a template
        return render_template('output.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
