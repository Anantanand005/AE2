from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

app = Flask(__name__)

# Generate some sample data and train a KMeans model for demonstration
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
model = KMeans(n_clusters=4)
model.fit(data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data point from request
        data_point = np.array([request.json['data']])
        prediction = model.predict(data_point)
        return jsonify({'cluster': int(prediction[0])})
    except KeyError as e:
        return jsonify({'error': f'Missing data for {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/centres', methods=['GET'])
def centres():
    try:
        centers = model.cluster_centers_
        return jsonify({'cluster_centers': centers.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data():
    try:
        # Assuming 'data' is the entire dataset loaded into the model
        df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
        df['Cluster'] = model.labels_
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)