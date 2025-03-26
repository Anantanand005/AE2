from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt
import json
import requests



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/altair")
def altair():
    return render_template("altair.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = scaler.transform([data["features"]])
    prediction = regressor.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})


@app.route("/clusters", methods=["GET"])
def clusters():
    centers = kmeans.cluster_centers_
    return jsonify({"cluster_centers": scaler.inverse_transform(centers).tolist()})


@app.route("/visualisation", methods=["GET"])
def visualisation():
    chart_data = df.copy()
    chart_data["Cluster"] = kmeans.labels_
    chart = (
        alt.Chart(chart_data)
        .mark_point()
        .encode(
            x="BMI", y="Y", color="Cluster:N", tooltip=["AGE", "SEX", "BMI", "BP", "Y"]
        )
        .interactive()
    )
    return render_template("chart.html", chart=chart.to_json())


@app.route("/hello")
def hello():
    return "<h1>Hello, Data Science World!</h1>"


@app.route("/diabetes")
def get_diabetes():
    print(request.args)
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

------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from flask import Flask, request, jsonify, render_template
from pickle import load
import altair as alt
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

with open("../AE2/chart.pkl", "rb") as f:
    data = load(f)

with open("../AE2/model.pkl", "rb") as f:
    model = load(f)

@app.route('/api/data', methods=['GET'])
def get_data():
    print(request.get_data())

    if not request.get_data():
        return {'error': 'Request body is missing'}, 400

    data = request.get_json()

    data = {
        "message": "Hello World!",
        "items": ["apple", "banana", "orange"]
    }
    return data

@app.route("/api/diabetes")
def get_diabetes():
    print("form data", request.args)
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
    return {
        "age": float(age),
        "sex": float(sex),
        "bmi": float(bmi),
        "bp": float(bp),
        "tc": float(tc),
        "ldl": float(ldl),
        "hdl": float(hdl),
        "tch": float(tch),
        "ltg": float(ltg),
        "glu": float(glu),
        "prediction": model.predict([age, sex, bmi, bp, tc, ldl, hdl, tch, ltg, glu]),
    }

features = ['AGE', 'SEX', 'BMI', 'BP', 'TC', 'LDL', 'HDL', 'TCH', 'LTG', 'GLU']

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return "Use POST with JSON body to get a prediction."
    data = request.json

    try:
        input_data = [[float(data[var]) for var in features]]  # Ensure correct type
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": prediction})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "All inputs must be numeric."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/visualisation", methods=["GET"])
def visualisation():
    df = pd.read_csv("diabetes.tab.tsv", sep="\t")
    df.rename(
        columns={
            "S1": "TC",
            "S2": "LDL",
            "S3": "HDL",
            "S4": "TCH",
            "S5": "LTG",
            "S6": "GLU",
        },
        inplace=True,
    )

    features = ["AGE", "SEX", "BMI", "BP", "TC", "LDL", "HDL", "TCH", "LTG", "GLU"]
    X = df[features].copy()
    y = df["Y"]

    X.fillna(X.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, shuffle=False
    )

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred_test = regressor.predict(X_test)

    train_df = pd.DataFrame({
        "BMI": X_train["BMI"].values,
        "Disease Progression": y_train,
        "Set": "Train",
        "Prediction": regressor.predict(X_train),
    })

    test_df = pd.DataFrame({
        "BMI": X_test["BMI"].values,
        "Disease Progression": y_test,
        "Set": "Test",
        "Prediction": y_pred_test,
    })

    full_df = pd.concat([train_df, test_df])

    base = alt.Chart(full_df).encode(
        x=alt.X("BMI", title="BMI"),
        y=alt.Y("Disease Progression", title="Disease Progression"),
        color=alt.Color("Set:N", legend=alt.Legend(title="Dataset Type")),
        tooltip=["BMI", "Disease Progression", "Set"],
    )

    points = base.mark_point()

    best_fit_line = base.transform_regression(
        "BMI", "Disease Progression", groupby=["Set"]
    ).mark_line()

    chart = (
        alt.layer(points, best_fit_line)
        .facet(column="Set:N")
        .properties(title="Linear Regression Analysis of Diabetes Progression Based on BMI")
    )

    chart_json = chart.to_json()

    return render_template("chart.html", chart=chart_json)

@app.route("/", methods=["GET"])
def form():
    return render_template("predict_form.html")
