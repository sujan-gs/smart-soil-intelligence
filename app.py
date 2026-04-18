import json
import os
import random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model_utils import get_dashboard_config, load_soil_model, predict_soil


app = Flask(__name__)
CORS(app)
soil_model = load_soil_model()

# Insert your Adafruit IO credentials here or set them as environment variables.
ADAFRUIT_IO_USERNAME = os.environ.get("ADAFRUIT_IO_USERNAME", "YOUR_ADAFRUIT_IO_USERNAME")
ADAFRUIT_IO_KEY = os.environ.get("ADAFRUIT_IO_KEY", "YOUR_ADAFRUIT_IO_KEY")
ADAFRUIT_FEEDS = {
    "moisture": os.environ.get("ADAFRUIT_FEED_MOISTURE", "moisture"),
    "ph": os.environ.get("ADAFRUIT_FEED_PH", "ph"),
    "ec": os.environ.get("ADAFRUIT_FEED_EC", "ec"),
    "temperature": os.environ.get("ADAFRUIT_FEED_TEMPERATURE", "temperature"),
    "nitrogen": os.environ.get("ADAFRUIT_FEED_NITROGEN", "nitrogen"),
    "phosphorus": os.environ.get("ADAFRUIT_FEED_PHOSPHORUS", "phosphorus"),
    "potassium": os.environ.get("ADAFRUIT_FEED_POTASSIUM", "potassium"),
    "rainfall": os.environ.get("ADAFRUIT_FEED_RAINFALL", "rainfall"),
}


def adafruit_configured():
    return (
        ADAFRUIT_IO_USERNAME
        and ADAFRUIT_IO_KEY
        and ADAFRUIT_IO_USERNAME != "farm100"
        and ADAFRUIT_IO_KEY != "aio_ueGI92Kjz18GgjsQZmm7rCaoi8XY"
    )


def demo_live_reading():
    return {
        "moisture": round(random.uniform(18, 82), 1),
        "ph": round(random.uniform(5.2, 8.2), 1),
        "ec": round(random.uniform(0.4, 2.8), 1),
        "temperature": round(random.uniform(20, 35), 1),
        "nitrogen": round(random.uniform(15, 120), 1),
        "phosphorus": round(random.uniform(8, 62), 1),
        "potassium": round(random.uniform(18, 95), 1),
        "rainfall": round(random.uniform(400, 1100), 1),
    }


def fetch_adafruit_feed(feed_key):
    url = f"https://io.adafruit.com/api/v2/{ADAFRUIT_IO_USERNAME}/feeds/{feed_key}/data/last"
    request_obj = Request(
        url,
        headers={
            "X-AIO-Key": ADAFRUIT_IO_KEY,
            "Accept": "application/json",
        },
    )
    with urlopen(request_obj, timeout=6) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return float(payload["value"])


def fetch_live_reading():
    if not adafruit_configured():
        return {"configured": False, "source": "simulated", "values": demo_live_reading()}

    try:
        values = {key: fetch_adafruit_feed(feed) for key, feed in ADAFRUIT_FEEDS.items()}
        return {"configured": True, "source": "adafruit", "values": values}
    except (HTTPError, URLError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Adafruit IO fetch failed: {exc}") from exc


@app.get("/")
def dashboard():
    return render_template("index.html")


@app.get("/api/health")
def health():
    accuracy = soil_model.get("training_accuracy") if isinstance(soil_model, dict) else None
    return jsonify({"status": "ok", "model_accuracy": accuracy, "adafruit_configured": adafruit_configured()})


@app.get("/api/config")
def config():
    config_payload = get_dashboard_config()
    config_payload["adafruit"] = {
        "configured": adafruit_configured(),
        "username": ADAFRUIT_IO_USERNAME if adafruit_configured() else None,
        "feeds": ADAFRUIT_FEEDS,
    }
    return jsonify(config_payload)


@app.get("/api/live-reading")
def live_reading():
    try:
        return jsonify(fetch_live_reading())
    except RuntimeError as exc:
        return jsonify({"configured": True, "source": "adafruit", "error": str(exc)}), 502


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_soil(payload, soil_model)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.post("/predict")
def predict_short_route():
    return predict()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
