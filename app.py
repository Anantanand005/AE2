from flask import Flask, render_template, request
import pandas as pd
from pickle import load

app = Flask(__name__)

with open("../AE2/chart.pkl", "rb") as f:
    chart = load(f)

with open("../AE2/model.pkl", "rb") as f:
    model = load(f)

@app.route('/')
def index():
    return render_template("answers.html")

@app.route("/diabetes", methods=["POST"])
def get_diabetes():
    print(request)
    print("form data", request.form.to_dict())

    age = request.form["AGE"]
    sex = request.form["SEX"]
    bmi = request.form["BMI"]
    bp = request.form["BP"]
    tc = request.form["TC"]
    ldl = request.form["LDL"]
    hdl = request.form["HDL"]
    tch = request.form["TCH"]
    ltg = request.form["LTG"]
    glu = request.form["GLU"]

    prediction = model.predict([[
        float(age),
        float(sex),
        float(bmi),
        float(bp),
        float(tc),
        float(ldl),
        float(hdl),
        float(tch),
        float(ltg),
        float(glu)
    ]])

    prediction = prediction[0]
    #return render_template("results.html", prediction=prediction, chart=chart.to_json())
    return {"prediction": prediction}

features = ['AGE', 'SEX', 'BMI', 'BP', 'TC', 'LDL', 'HDL', 'TCH', 'LTG', 'GLU']

@app.route("/api/diabetes")
def api_diabetes():
    age = request.args.get("AGE")
    sex = request.args.get("SEX")
    bmi = request.args.get("BMI")
    bp = request.args.get("BP")
    tc = request.args.get("TC")
    ldl = request.args.get("LDL")
    hdl = request.args.get("HDL")
    tch = request.args.get("tch")
    ltg = request.args.get("Ltg")
    glu = request.args.get("glu")

    prediction = model.predict([[
        float(age),
        float(sex),
        float(bmi),
        float(bp),
        float(tc),
        float(ldl),
        float(hdl),
        float(tch),
        float(ltg),
        float(glu)
    ]])

    return {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "bp": bp,
        "tc": tc,
        "ldl": ldl,
        "hdl": hdl,
        "tch": tch,
        "ltg": ltg,
        "glu": glu,
        "prediction": model.predict([age, sex, bmi, bp, tc, ldl, hdl, tch, ltg, glu]),
    }