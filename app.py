
#####################################################################
#### app.py ####


from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

app = Flask(__name__)

# Dataset
data = {
    'Yobs': [137, 118, 124, 124, 120, 129, 122, 142, 128, 114,
             132, 130, 130, 112, 132, 117, 134, 132, 121, 128],
    'W':    [0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
             1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    'X':    [19.8, 23.4, 27.7, 24.6, 21.5, 25.1, 22.4, 29.3, 20.8, 20.2,
             27.3, 24.5, 22.9, 18.4, 24.2, 21.0, 25.9, 23.2, 21.6, 22.8]
}

df = pd.DataFrame(data)

# Model with control (W and X)
X_train = df[['W', 'X']]
y_train = df['Yobs']

model = LinearRegression().fit(X_train, y_train)

@app.route("/predict")
def predict():
    try:
        W = float(request.args.get("W"))
        X = float(request.args.get("X"))
    except (TypeError, ValueError):
        return jsonify({"error": "Please provide valid numeric values for W and X"}), 400

    y_pred = model.predict([[W, X]])[0]

    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input W: {W}, X: {X}\nPrediction: {y_pred}\n")

    return jsonify({"W": W, "X": X, "predicted_Yobs": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)







