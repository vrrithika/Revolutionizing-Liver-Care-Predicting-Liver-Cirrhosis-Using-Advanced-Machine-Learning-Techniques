from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import traceback

# Load preprocessing objects and model
model = joblib.load("liver_cirosis/logreg_liver_cirosis_model.pkl")

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        # Retrieve form data
        Age = float(request.form["Age"])
        Quantity_of_alcohol_consumption = float(request.form["Quantity_of_alcohol_consumption"])
        Diabetes_Result = request.form["Diabetes_Result"]
        Blood_pressure = request.form["Blood_pressure"]
        Hemoglobin = float(request.form["Hemoglobin"])
        PCV = float(request.form["PCV"])
        Polymorphs = float(request.form["Polymorphs"])
        Lymphocytes = float(request.form["Lymphocytes"])
        Platelet_Count = float(request.form["Platelet_Count"])
        Indirect = float(request.form["Indirect"])
        Total_Protein = float(request.form["Total_Protein"])
        Albumin = float(request.form["Albumin"])
        Globulin = float(request.form["Globulin"])
        AG_Ratio = float(request.form["AG_Ratio"])
        AL_Phosphatase = float(request.form["AL_Phosphatase"])
        USG_Abdomen = request.form["USG_Abdomen"]

        # Convert categorical inputs to numerical
        Diabetes_Result = 1 if Diabetes_Result.lower() == "yes" else 0
        USG_Abdomen = 1 if USG_Abdomen.lower() == "yes" else 0

        # Handle blood pressure input
        systolic_pressure = float(Blood_pressure.split('/')[0]) / float(Blood_pressure.split('/')[1])

        # Create input array for the model
        x_test = [[Age, Quantity_of_alcohol_consumption, Diabetes_Result, systolic_pressure,
                   Hemoglobin, PCV, Polymorphs, Lymphocytes,
                   Platelet_Count, Indirect, Total_Protein, Albumin,
                   Globulin, AG_Ratio, AL_Phosphatase, USG_Abdomen]]

        # Make prediction
        prediction = model.predict(x_test)

        # Determine prediction text
        if prediction == 0:
            prediction_text = "No liver cirrhosis"
        else:
            prediction_text = "Liver cirrhosis detected"

        return jsonify({"prediction_text": prediction_text})
    except Exception as e:
        error_message = f"Error during prediction: {e}"
        print(error_message)
        print(traceback.format_exc())
        return jsonify({"prediction_text": error_message})

if __name__ == "__main__":
    app.run(debug=True)
