from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector, joblib
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import joblib


app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='engine'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


df = pd.read_csv('newengine_data.csv')
X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    

@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    if request.method == "POST":
        algorithm_type = request.form['algorithm']
        accuracy = None
        if algorithm_type == 'Decision Tree Classifier':
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_train, y_train)
            y_pred_dt = dt_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_dt)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)
        
        elif algorithm_type == 'Random Forest Classifier':
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_rf)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy) 
        
        elif algorithm_type == 'Logistic Regression':
            lr_model = LogisticRegression(random_state=42)
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_lr)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)
        
        elif algorithm_type == 'K Nearest Nieghbo':
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train, y_train)
            y_pred_knn = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_knn)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)  
        
        elif algorithm_type == 'Linear Disciminent Analysis':
            lda_model = LinearDiscriminantAnalysis()
            lda_model.fit(X_train, y_train)
            y_pred_lda = lda_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_lda)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)
        
        elif algorithm_type == 'Adaboost Classifier':
            ada_model = AdaBoostClassifier(random_state=42)
            ada_model.fit(X_train, y_train)
            y_pred_ada = ada_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_ada)
            accuracy = str(accuracy)[2:4]
            return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)
    return render_template('algorithm.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Retrieve input values from the form
        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])

        # Load the saved scaler and models
        scaler = joblib.load('scaler.joblib')
        rf_model = joblib.load('rf_model.joblib')
        dt_model = joblib.load('dt_model.joblib')
        knn_model = joblib.load('knn_model.joblib')

        # Prepare and standardize input data for prediction
        input_data = np.array([[a, b, c, d, e, f]])
        input_data_scaled = scaler.transform(input_data)

        # Make predictions using the loaded models
        rf_prediction = rf_model.predict(input_data_scaled)
        dt_prediction = dt_model.predict(input_data_scaled)
        knn_prediction = knn_model.predict(input_data_scaled)

        # Combine predictions (e.g., majority voting)
        predictions = [rf_prediction[0], dt_prediction[0], knn_prediction[0]]
        final_prediction = max(set(predictions), key=predictions.count)

        # Convert prediction label to corresponding class
        predicted_class = 'Engine Health is Good' if final_prediction == 1 else 'Engine Health is Bad.'

        # Render prediction.html template with prediction value
        return render_template('prediction.html', prediction=predicted_class)

    return render_template('prediction.html')



@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)

