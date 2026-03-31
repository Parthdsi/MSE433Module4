import os
import csv
from datetime import datetime
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")

model1_data = joblib.load(os.path.join(MODEL_DIR, "ep_model1.pkl"))
model2_data = joblib.load(os.path.join(MODEL_DIR, "ep_model2.pkl"))

MODEL1 = model1_data["model"]
MODEL1_FEATURES = model1_data["feature_names"]

MODEL2 = model2_data["model"]
MODEL2_FEATURES = model2_data["feature_names"]

COMPLEXITY_MAP = {
    "Standard PVI": 0,
    "BOX or PST BOX": 1,
    "CTI or SVC": 2,
    "AAFL": 3,
}

PHYSICIAN_OPTIONS = ["Dr. A", "Dr. B", "Dr. C"]


def build_feature_vector(feature_names, values_dict):
    vec = []
    for fname in feature_names:
        vec.append(float(values_dict.get(fname, 0.0)))
    return np.array([vec])


@app.route("/predict/model1", methods=["POST"])
def predict_model1():
    data = request.get_json()
    physician = data.get("physician", "Dr. A")
    case_of_day = int(data.get("case_of_day", 1))
    complexity = data.get("complexity", "Standard PVI")

    values = {
        "case_of_day": case_of_day,
        "complexity_tier": COMPLEXITY_MAP.get(complexity, 0),
    }

    for doc in PHYSICIAN_OPTIONS:
        col = f"physician_{doc}"
        if col in MODEL1_FEATURES:
            values[col] = 1.0 if physician == doc else 0.0

    X = build_feature_vector(MODEL1_FEATURES, values)
    pred = float(MODEL1.predict(X)[0])

    return jsonify({"predicted_minutes": round(pred, 1), "model": "model1"})


@app.route("/predict/model2", methods=["POST"])
def predict_model2():
    data = request.get_json()
    physician = data.get("physician", "Dr. A")
    case_of_day = int(data.get("case_of_day", 1))
    complexity = data.get("complexity", "Standard PVI")

    values = {
        "case_of_day": case_of_day,
        "complexity_tier": COMPLEXITY_MAP.get(complexity, 0),
        "first_case_of_day_enc": 1.0 if data.get("first_case_of_day", "N") == "Y" else 0.0,
        "obesity_enc": 1.0 if data.get("obesity", "N") == "Y" else 0.0,
        "sleep_apnea_enc": 1.0 if data.get("sleep_apnea", "N") == "Y" else 0.0,
        "fasting_not_confirmed_enc": 1.0 if data.get("fasting_not_confirmed", "N") == "Y" else 0.0,
        "bloodwork_incomplete_enc": 1.0 if data.get("bloodwork_incomplete", "N") == "Y" else 0.0,
        "equipment_prestaged_enc": 1.0 if data.get("equipment_prestaged", "Y") == "Y" else 0.0,
        "anesthesia_ready_enc": 1.0 if data.get("anesthesia_ready", "Y") == "Y" else 0.0,
    }

    for doc in PHYSICIAN_OPTIONS:
        col = f"physician_{doc}"
        if col in MODEL2_FEATURES:
            values[col] = 1.0 if physician == doc else 0.0

    X = build_feature_vector(MODEL2_FEATURES, values)
    pred = float(MODEL2.predict(X)[0])

    return jsonify({"predicted_minutes": round(pred, 1), "model": "model2"})


LOG_CSV = os.path.join(MODEL_DIR, "logged_cases.csv")

LOG_COLUMNS = [
    "CASE #", "DATE", "PHYSICIAN", "PT PREP/INTUBATION", "ACCESSS", "TSP",
    "PRE-MAP", "ABL DURATION", "ABL TIME", "#ABL", "LA DWELL TIME",
    "CASE TIME", "POST CARE/EXTUBATION", "PT IN-OUT", "Note",
    "first_case_of_day", "obesity", "sleep_apnea", "fasting_not_confirmed",
    "bloodwork_incomplete", "equipment_prestaged", "anesthesia_ready",
    "tsp_difficulty", "arterial_line_time_min", "equipment_issue",
    "equipment_issue_detail", "patient_out_time", "pacu_called_time",
    "cleaning_start_time", "cleaning_end_time", "cleaning_duration_min",
    "delay_reason",
]

FORM_TO_CSV = {
    "case_number": "CASE #",
    "date": "DATE",
    "physician": "PHYSICIAN",
    "pt_prep": "PT PREP/INTUBATION",
    "access": "ACCESSS",
    "tsp": "TSP",
    "pre_map": "PRE-MAP",
    "abl_duration": "ABL DURATION",
    "abl_time": "ABL TIME",
    "num_abl": "#ABL",
    "la_dwell_time": "LA DWELL TIME",
    "case_time": "CASE TIME",
    "post_care": "POST CARE/EXTUBATION",
    "pt_in_out": "PT IN-OUT",
    "note": "Note",
    "first_case_of_day": "first_case_of_day",
    "obesity": "obesity",
    "sleep_apnea": "sleep_apnea",
    "fasting_not_confirmed": "fasting_not_confirmed",
    "bloodwork_incomplete": "bloodwork_incomplete",
    "equipment_prestaged": "equipment_prestaged",
    "anesthesia_ready": "anesthesia_ready",
    "tsp_difficulty": "tsp_difficulty",
    "arterial_line_time_min": "arterial_line_time_min",
    "equipment_issue": "equipment_issue",
    "equipment_issue_detail": "equipment_issue_detail",
    "patient_out_time": "patient_out_time",
    "pacu_called_time": "pacu_called_time",
    "cleaning_start_time": "cleaning_start_time",
    "cleaning_end_time": "cleaning_end_time",
    "cleaning_duration_min": "cleaning_duration_min",
    "delay_reason": "delay_reason",
}


@app.route("/log", methods=["POST"])
def log_case():
    data = request.get_json()
    write_header = not os.path.exists(LOG_CSV)

    row = {}
    for form_key, csv_col in FORM_TO_CSV.items():
        row[csv_col] = data.get(form_key, "")

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return jsonify({"status": "saved", "case": row.get("CASE #", "")})


@app.route("/log/count", methods=["GET"])
def log_count():
    if not os.path.exists(LOG_CSV):
        return jsonify({"count": 0})
    with open(LOG_CSV, "r") as f:
        count = sum(1 for _ in f) - 1
    return jsonify({"count": max(count, 0)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model1_features": MODEL1_FEATURES,
        "model2_features": MODEL2_FEATURES,
    })


if __name__ == "__main__":
    print(f"Model 1 features: {MODEL1_FEATURES}")
    print(f"Model 2 features: {MODEL2_FEATURES}")
    app.run(host="0.0.0.0", port=5001, debug=True)
