import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) # Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # Load the trained model

category_mapping = {
    'Male': 0,
    'Female': 1,
    'never': 2,
    'No Info': 3,
    'current': 4,
    'former': 5,
    'ever': 6,
    'not current': 7
}  
@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    init_features = [category_mapping.get(x, x) for x in request.form.values()]
    init_features = [float(x) for x in init_features]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # Make a prediction
    
    if prediction == 0:
        return render_template('index.html', prediction_text='Diabetes Prediction: No diabetes')
    else:
        return render_template('index.html', prediction_text='Diabetes Prediction: you have diabetes')

if __name__ == "__main__":
    app.run(debug=True)