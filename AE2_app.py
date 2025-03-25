from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt
import json

app = Flask(__name__)

# Assuming the data is preloaded and preprocessed
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
X = df[features]
y = df["Y"]
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_scaled, y)

# Fit a KMeans clustering model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)


@app.route("/")
def index():
    return render_template("index.html")


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/altair")
def altair():
    return render_template("altair.html")


@app.route("/hello")
def hello():
    return "<h1>Hello, Data Science World!</h1>"


if __name__ == "__main__":
    app.run(debug=True)
