from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import os
from flask_cors import CORS
import sklearn
import numpy as np
print(f"Using scikit-learn version: {sklearn.__version__}") 

# Load trained model pipeline from the correct location
MODEL_PATH = Path("model_artifacts_mid_ipynb/best_model_mid.pkl")

# Check if the new model exists, otherwise fall back to root directory
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
elif os.path.exists("best_model_mid.pkl"):
    print("WARNING: Using old model file. Please retrain with the notebook to avoid compatibility issues.")
    model = joblib.load("best_model_mid.pkl")
else:
    raise FileNotFoundError("No model file found! Please run the training notebook first.")

# Get feature list from the model (more robust than hardcoding)
try:
    FEATURES = list(model.feature_names_in_)
    print(f"Using {len(FEATURES)} features from trained model")
except AttributeError:
    # Fallback for older models that might not have this attribute
    FEATURES = [
        "fx_lkr_per_usd_m", "kenya_bopf_price_usd_w", "india_bopf_price_usd_w",
        "fob_rs_per_kg_wavg_m", "rain_mm_sum_w", "temp_mean_c_w", "month",
        "bopf_price_lkr_per_kg_lag1", "bopf_price_lkr_per_kg_lag4", "bopf_price_lkr_per_kg_lag8",
        "fx_lkr_per_usd_m_lag1","fx_lkr_per_usd_m_lag4","fx_lkr_per_usd_m_lag8",
        "kenya_bopf_price_usd_w_lag1","kenya_bopf_price_usd_w_lag4","kenya_bopf_price_usd_w_lag8",
        "india_bopf_price_usd_w_lag1","india_bopf_price_usd_w_lag4","india_bopf_price_usd_w_lag8",
        "rain_4w_sum","price_ma4w"
    ]
    print("Using fallback feature list")

app = Flask(__name__)
CORS(app)  # <-- allow all origins

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json(force=True)

        # ---- 1) Load recent historical data ----
        df = pd.read_csv("weekly_model_table.csv", parse_dates=["auction_date_start"])
        # robust elevation parse
        elev = df["elevation"].astype(str).str.lower()
        df["elev_norm"] = np.where(elev.str.contains("high"), "High",
                            np.where(elev.str.contains("mid|medium"), "Mid", "Low"))
        df_mid = df[df["elev_norm"] == "Mid"].copy().sort_values("auction_date_start")

        # last 12 weeks WITH competitor columns so frontend can draw dotted lines directly
        history_cols = [
            "auction_date_start",
            "bopf_price_lkr_per_kg",
            "kenya_bopf_price_usd_w",
            "india_bopf_price_usd_w",
        ]
        history = (
            df_mid.loc[:, history_cols]
                  .tail(12)
                  .assign(
                      # ensure pure Python types for JSON
                      bopf_price_lkr_per_kg=lambda x: x["bopf_price_lkr_per_kg"].astype(float).round(2),
                      kenya_bopf_price_usd_w=lambda x: x["kenya_bopf_price_usd_w"].astype(float).round(2),
                      india_bopf_price_usd_w=lambda x: x["india_bopf_price_usd_w"].astype(float).round(2),
                  )
        )

        # ---- 2) Prepare input for forecast ----
        X = pd.DataFrame([[data.get(f, 0) for f in FEATURES]], columns=FEATURES)
        forecast_price = float(model.predict(X)[0])

        # next timestamp = last date + 7 days
        next_date = pd.to_datetime(history["auction_date_start"].max()) + pd.Timedelta(days=7)

        # Simple CI: max(±30 LKR, half of recent 8w std)
        recent_std = float(history["bopf_price_lkr_per_kg"].tail(8).std() or 0)
        ci_half = max(30.0, 0.5 * recent_std)
        ci_lower = round(forecast_price - ci_half, 2)
        ci_upper = round(forecast_price + ci_half, 2)

        # ---- 3) Build response ----
        return jsonify({
            # History now includes kenya/india on each row (frontend reads d.kenya / d.india directly)
            "history": [
                {
                    "auction_date_start": str(r["auction_date_start"].date()),
                    "bopf_price_lkr_per_kg": float(r["bopf_price_lkr_per_kg"]),
                    "kenya_bopf_price_usd_w": (None if pd.isna(r["kenya_bopf_price_usd_w"]) else float(r["kenya_bopf_price_usd_w"])),
                    "india_bopf_price_usd_w": (None if pd.isna(r["india_bopf_price_usd_w"]) else float(r["india_bopf_price_usd_w"])),
                }
                for _, r in history.iterrows()
            ],
            # Keep these for backward compatibility (the React you have doesn’t need them anymore)
            "kenya_history": [],
            "india_history": [],
            "forecast": {
                "auction_date_start": str(next_date.date()),
                "forecast_price_lkr": round(forecast_price, 2),
                "confidence": f"±{int(round(ci_half))} LKR (estimated)",
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        })

    except Exception as e:
        print(f"Error during forecasting: {e}")
        return jsonify({"error": str(e)}), 400



@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Tea Price Forecast API is running",
        "endpoints": {
            "/forecast": "POST - Make a price prediction",
            "/features": "GET - Get list of required features"
        }
    })

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({
        "required_features": FEATURES,
        "feature_count": len(FEATURES),
        "example_request": {
            "url": "/forecast",
            "method": "POST",
            "body": {f: f"<{f}_value>" for f in FEATURES[:5]}
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
