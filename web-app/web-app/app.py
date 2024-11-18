""" import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("ufo-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )

if __name__ == "__main__":
    app.run(debug=True)"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

app = Flask(__name__)

# Initialize the model as none, as it will be trained mannualy
model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    # Set a condition in case the model is not trained yet
    if model is None:
        return render_template("index.html", prediction_text="Model not trained yet. Please train the model first.")

    try:
        # Extract input features from the form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = prediction[0]

        countries = ["Australia", "Canada", "Germany", "UK", "US"]

        return render_template("index.html", prediction_text="Likely country: {}".format(countries[output]))
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

# Add the train rout to train the model mannualy
@app.route("/train", methods=["GET"])
def train():
    global model

    try:
        # Load and preprocess data
        ufos = pd.read_csv("../data/ufos.csv")
        ufos = pd.DataFrame({
            'Seconds': ufos['duration (seconds)'], 
            'Country': ufos['country'],
            'Latitude': ufos['latitude'],
            'Longitude': ufos['longitude']
        })
        ufos.dropna(inplace=True)

        # Encode the country labels
        ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

        # Filter seconds between 1 and 60
        ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

        # Select features and split data
        selected_features = ['Seconds', 'Latitude', 'Longitude']
        X = ufos[selected_features]
        y = ufos['Country']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Pickle the model
        model_filename = 'ufo-model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, output_dict=True)

        return jsonify({
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "classification_report": classification
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
