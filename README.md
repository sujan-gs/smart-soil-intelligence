# Smart Soil Intelligence System

A hackathon-ready local prototype for AI-driven soil analysis and precision farming.

The app accepts soil sensor readings for moisture, pH, EC, temperature, nitrogen, phosphorus, and potassium, then returns a fertilizer recommendation, irrigation suggestion, and per-parameter health statuses.

The dashboard also includes a language selector and a receipt-style PDF flow. After a prediction, click **Download PDF Receipt**, then choose **Save as PDF** in the browser print window.

## Project Structure

```text
.
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── data/
│   └── soil_data.csv
├── models/
│   └── soil_model.pkl        # created after training
├── static/
│   ├── script.js
│   └── styles.css
└── templates/
    └── index.html
```

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open the dashboard at:

```text
http://127.0.0.1:5000
```

## API

`POST /api/predict`

Example request:

```json
{
  "moisture": 42,
  "ph": 6.7,
  "ec": 1.1,
  "temperature": 27,
  "nitrogen": 70,
  "phosphorus": 35,
  "potassium": 55
}
```

Example response:

```json
{
  "recommendation": "Balanced soil - maintain current fertilizer plan",
  "irrigation": "Moisture is adequate. Keep normal irrigation schedule.",
  "statuses": {
    "moisture": "Optimal",
    "ph": "Optimal",
    "ec": "Optimal",
    "temperature": "Optimal",
    "nitrogen": "Optimal",
    "phosphorus": "Optimal",
    "potassium": "Optimal"
  },
  "overall_status": "Optimal",
  "confidence": 0.74
}
```
