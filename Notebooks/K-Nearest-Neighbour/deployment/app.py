from flask import Flask, request, render_template

import pandas as pd
import pickle

loaded_knn_model = pickle.load(open("../knn_model_file", "rb"))

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def hello_world():
    if request.method == "POST":
        try:
            inputs = {}
            
            inputs["Age"] = [int(request.form["age"])]
            inputs["Height(cm)"] = [int(request.form["height"])]
            inputs["Weight(kg)"] = [int(request.form["weight"])]
            inputs["Cholesterol(mg/dL)"] = [int(request.form["cholesterol"])]
            inputs["Glucose(mg/dL)"] = [int(request.form["glucose"])]
            inputs["Smoker"] = [request.form.get("smoker") == "on" and 1 or 0]
            inputs["Exercise(hours/week)"] = [int(request.form["exercise"])]
            inputs["Systolic"] = [int(request.form["systolic"])]
            inputs["Diastolic"] = [int(request.form["diastolic"])]
            
            new_df = pd.DataFrame(inputs)
            prediction = loaded_knn_model.predict(new_df) == [1]
            return render_template("index.html", prediction=prediction)
        except Exception as e:
            print("Error: ", e)
            return render_template("index.html")
    else:
        return render_template("index.html")
