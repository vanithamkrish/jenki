from flask import Flask, request, render_template
import pickle

# Step 1: Create Flask App
app = Flask(__name__)

# Step 2: Load Trained Model
model = pickle.load(open('model.pkl', 'rb'))

# Step 3: Route for Home Page (Form)
@app.route('/')
def home():
    return render_template('index.html')

# Step 4: Route for Prediction
@app.route('/predict_file', methods=['POST'])
def predict_file():
    # Extract features from form input
    features = [[
        float(request.form[f]) for f in [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]
    ]]
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return result back to the HTML page
    return render_template(
        'index.html',
        prediction_text=f'Prediction: {"Heart Disease Detected" if prediction[0]==1 else "No Heart Disease"}'
    )

# Step 5: Run the App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
