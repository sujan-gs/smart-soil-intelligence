from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "soil_model.pkl"
FEATURE_COLUMNS = [
    "moisture",
    "ph",
    "ec",
    "temperature",
    "nitrogen",
    "phosphorus",
    "potassium",
    "rainfall",
    "district",
]

DISTRICT_PROFILES = {
    "Mandya": {
        "kn": "ಮಂಡ್ಯ",
        "avg_rainfall": 720,
        "note": "Canal-fed belt with strong paddy and sugarcane potential.",
    },
    "Mysore": {
        "kn": "ಮೈಸೂರು",
        "avg_rainfall": 760,
        "note": "Balanced climate suitable for paddy, tobacco, pulses, and vegetables.",
    },
    "Davangere": {
        "kn": "ದಾವಣಗೆರೆ",
        "avg_rainfall": 640,
        "note": "Maize, millets, and cotton fit well in moderately dry belts.",
    },
    "Belagavi": {
        "kn": "ಬೆಳಗಾವಿ",
        "avg_rainfall": 810,
        "note": "Diverse soils support sugarcane, maize, wheat, and vegetables.",
    },
    "Shivamogga": {
        "kn": "ಶಿವಮೊಗ್ಗ",
        "avg_rainfall": 1280,
        "note": "High-rainfall zone that favors paddy and water-demanding crops.",
    },
    "Tumakuru": {
        "kn": "ತುಮಕೂರು",
        "avg_rainfall": 690,
        "note": "Millets, groundnut, and pulses perform well under careful moisture management.",
    },
    "Raichur": {
        "kn": "ರಾಯಚೂರು",
        "avg_rainfall": 590,
        "note": "Warm semi-arid district suitable for cotton, paddy pockets, and pulses.",
    },
    "Vijayapura": {
        "kn": "ವಿಜಯಪುರ",
        "avg_rainfall": 560,
        "note": "Low-rainfall region where drought-tolerant crops are safer.",
    },
    "Kolar": {
        "kn": "ಕೋಲಾರ",
        "avg_rainfall": 700,
        "note": "Groundnut, millets, oil seeds, and vegetables are common choices.",
    },
    "Hassan": {
        "kn": "ಹಾಸನ",
        "avg_rainfall": 1040,
        "note": "Higher rainfall and moderate temperatures support maize and sugarcane.",
    },
}

CROP_LIBRARY = {
    "Rice": {
        "kn": "ಅಕ್ಕಿ",
        "dataset_label": "Rice",
        "requirements": {"nitrogen": 100, "phosphorus": 50, "potassium": 50},
        "rotation": "Follow with pulses or groundnut to recover soil nitrogen.",
        "income": "Use short-duration varieties, schedule irrigation, and sell through local mandis or FPOs.",
    },
    "Maize": {
        "kn": "ಜೋಳ (ಮೆಕ್ಕೆಜೋಳ)",
        "dataset_label": "Maize",
        "requirements": {"nitrogen": 150, "phosphorus": 60, "potassium": 50},
        "rotation": "Rotate with pulses to improve nitrogen balance and reduce pest pressure.",
        "income": "Bundle grain with fodder residue sales for stronger farm returns.",
    },
    "Wheat": {
        "kn": "ಗೋಧಿ",
        "dataset_label": "Wheat",
        "requirements": {"nitrogen": 120, "phosphorus": 60, "potassium": 40},
        "rotation": "Rotate with oil seeds or pulses after harvest.",
        "income": "Split nitrogen doses and use certified seed for steadier yield.",
    },
    "Millets": {
        "kn": "ಸಿರಿಧಾನ್ಯಗಳು",
        "dataset_label": "Millets",
        "requirements": {"nitrogen": 80, "phosphorus": 40, "potassium": 40},
        "rotation": "Rotate with pulses or vegetables when irrigation improves.",
        "income": "Millets reduce water cost and fit well in low-risk dryland systems.",
    },
    "Pulses": {
        "kn": "ಕಾಳು ಬೆಳೆಗಳು",
        "dataset_label": "Pulses",
        "requirements": {"nitrogen": 25, "phosphorus": 50, "potassium": 25},
        "rotation": "Excellent rotation crop after cereals because it improves soil nitrogen.",
        "income": "Intercrop pulses to reduce fertilizer cost and create an extra harvest stream.",
    },
    "Sugarcane": {
        "kn": "ಕರಿಬೇವು?",
        "dataset_label": "Sugarcane",
        "requirements": {"nitrogen": 250, "phosphorus": 100, "potassium": 120},
        "rotation": "Rotate with pulses or green manure before the next cane cycle.",
        "income": "Use ratoon management only after correcting nutrient gaps and maintaining moisture.",
    },
    "Groundnut": {
        "kn": "ಕಡಲೆಕಾಯಿ",
        "dataset_label": "Groundnut",
        "requirements": {"nitrogen": 20, "phosphorus": 40, "potassium": 40},
        "rotation": "Rotate with millets or maize to diversify income and break disease cycles.",
        "income": "Groundnut works well in medium moisture fields with good market access.",
    },
    "Cotton": {
        "kn": "ಹತ್ತಿ",
        "dataset_label": "Cotton",
        "requirements": {"nitrogen": 120, "phosphorus": 60, "potassium": 60},
        "rotation": "Rotate with pulses to reduce nutrient mining and pest carryover.",
        "income": "Improve lint quality by correcting potassium and avoiding water stress during flowering.",
    },
    "Oil Seeds": {
        "kn": "ಎಣ್ಣೆ ಬೀಜಗಳು",
        "dataset_label": "Oil Seeds",
        "requirements": {"nitrogen": 60, "phosphorus": 40, "potassium": 40},
        "rotation": "Rotate with cereals or pulses for better field balance.",
        "income": "Oil seeds fit low to medium rainfall belts and reduce irrigation risk.",
    },
    "Tobacco": {
        "kn": "ತಂಬಾಕು",
        "dataset_label": "Tobacco",
        "requirements": {"nitrogen": 80, "phosphorus": 60, "potassium": 80},
        "rotation": "Rotate with pulses or vegetables to soften disease pressure.",
        "income": "Only continue tobacco if market access is secured and salinity remains under control.",
    },
    "Barley": {
        "kn": "ಜವ",
        "dataset_label": "Barley",
        "requirements": {"nitrogen": 60, "phosphorus": 30, "potassium": 30},
        "rotation": "Rotate with pulses after harvest to rebuild fertility.",
        "income": "Barley can be a lower-risk cereal when temperatures stay moderate.",
    },
}

# The Kannada crop name for Sugarcane is intentionally simple for demo readability.
CROP_LIBRARY["Sugarcane"]["kn"] = "ಕರಿಬೇಲು"

FEATURE_LABELS = {
    "moisture": {"en": "Moisture", "kn": "ತೇವಾಂಶ"},
    "ph": {"en": "pH", "kn": "ಪಿಎಚ್"},
    "ec": {"en": "EC", "kn": "ಇಸಿ"},
    "temperature": {"en": "Temperature", "kn": "ತಾಪಮಾನ"},
    "nitrogen": {"en": "Nitrogen", "kn": "ನೈಟ್ರೋಜನ್"},
    "phosphorus": {"en": "Phosphorus", "kn": "ಫಾಸ್ಫರಸ್"},
    "potassium": {"en": "Potassium", "kn": "ಪೊಟ್ಯಾಸಿಯಂ"},
    "rainfall": {"en": "Rainfall", "kn": "ಮಳೆಯ ಪ್ರಮಾಣ"},
}

THRESHOLDS = {
    "moisture": (25, 65),
    "ph": (6.0, 7.5),
    "ec": (0.4, 1.8),
    "temperature": (18, 32),
    "nitrogen": (40, 90),
    "phosphorus": (20, 45),
    "potassium": (35, 80),
    "rainfall": (550, 1000),
}

STATUS_FACTORS = {"Low": 0.25, "Medium": 0.5, "High": 0.75}
STATUS_TRANSLATIONS = {"Low": "ಕಡಿಮೆ", "Medium": "ಮಧ್ಯಮ", "High": "ಹೆಚ್ಚು"}
ISSUE_PRIORITY = ["nitrogen", "phosphorus", "potassium", "moisture", "ec", "ph", "rainfall", "temperature"]

DEFICIENCY_LIBRARY = {
    "nitrogen": {
        "en": "Nitrogen is low. Watch for yellow older leaves and weak vegetative growth.",
        "kn": "ನೈಟ್ರೋಜನ್ ಕಡಿಮೆಯಾಗಿದೆ. ಹಳೆಯ ಎಲೆಗಳು ಹಳದಿ ಬಣ್ಣವಾಗುವುದು ಮತ್ತು ಬೆಳವಣಿಗೆ ಕುಂದುವುದು ಕಾಣಬಹುದು.",
    },
    "phosphorus": {
        "en": "Phosphorus is low. Root growth may weaken and leaves can show dull purple shades.",
        "kn": "ಫಾಸ್ಫರಸ್ ಕಡಿಮೆಯಾಗಿದೆ. ಬೇರು ಬೆಳವಣಿಗೆ ಕುಂದಬಹುದು ಮತ್ತು ಎಲೆಗಳಲ್ಲಿ ನೇರಳೆ ಮಸುಕಿನ ಲಕ್ಷಣ ಕಾಣಬಹುದು.",
    },
    "potassium": {
        "en": "Potassium is low. Leaf edge scorching and poor stress tolerance may appear.",
        "kn": "ಪೊಟ್ಯಾಸಿಯಂ ಕಡಿಮೆಯಾಗಿದೆ. ಎಲೆ ಅಂಚು ಸುಡುವುದು ಮತ್ತು ಒತ್ತಡ ಸಹನ ಶಕ್ತಿ ಕಡಿಮೆಯಾಗುವುದು ಕಾಣಬಹುದು.",
    },
}

IRRIGATION_TEMPLATES = {
    "urgent": {
        "en": "Moisture is low for this field. Give a light irrigation immediately and follow with 25-35 mm based on crop stage.",
        "kn": "ಈ ಜಮೀನಿನ ತೇವಾಂಶ ಕಡಿಮೆ ಇದೆ. ತಕ್ಷಣ ಸಣ್ಣ ಪ್ರಮಾಣದ ನೀರಾವರಿ ನೀಡಿ, ನಂತರ ಬೆಳೆ ಹಂತಕ್ಕೆ ಅನುಗುಣವಾಗಿ 25-35 ಮಿಮೀ ನೀರು ನೀಡಿ.",
    },
    "balanced": {
        "en": "Moisture is acceptable. Maintain the normal irrigation schedule and avoid overwatering.",
        "kn": "ತೇವಾಂಶ ಸಮತೋಲನದಲ್ಲಿದೆ. ಸಾಮಾನ್ಯ ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿಯನ್ನು ಮುಂದುವರಿಸಿ ಮತ್ತು ಅತಿಯಾಗಿ ನೀರು ಹಾಕಬೇಡಿ.",
    },
    "drainage": {
        "en": "The field is already wet. Reduce irrigation and improve drainage to avoid root stress and salinity build-up.",
        "kn": "ಜಮೀನು ಈಗಾಗಲೇ ಒದ್ದೆಯಾಗಿದೆ. ನೀರಾವರಿಯನ್ನು ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಬೇರು ಒತ್ತಡ ಹಾಗೂ ಉಪ್ಪುದ್ರವ್ಯ ಏರಿಕೆಯನ್ನು ತಪ್ಪಿಸಲು ನೀರು ಹೊರಹಾಕುವ ವ್ಯವಸ್ಥೆ ಸುಧಾರಿಸಿ.",
    },
}

INCOME_TEMPLATES = {
    "low_soil": {
        "en": "Correct the main soil limits first, then prioritize lower-risk crops or intercropping to protect net income.",
        "kn": "ಮೊದಲು ಮಣ್ಣಿನ ಪ್ರಮುಖ ಸಮಸ್ಯೆಗಳನ್ನು ಸರಿಪಡಿಸಿ, ನಂತರ ಕಡಿಮೆ ಅಪಾಯದ ಬೆಳೆಗಳು ಅಥವಾ ಅಂತರ ಬೆಳೆ ಪದ್ಧತಿಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಆದಾಯವನ್ನು ಸುರಕ್ಷಿತಗೊಳಿಸಿ.",
    },
    "rice": {
        "en": "Use scheduled irrigation, shorter-duration seed, and direct procurement channels to improve rice margins.",
        "kn": "ನಿಯೋಜಿತ ನೀರಾವರಿ, ಕಡಿಮೆ ಅವಧಿಯ ಬೀಜ, ಮತ್ತು ನೇರ ಖರೀದಿ ಮಾರ್ಗಗಳನ್ನು ಬಳಸುವುದರಿಂದ ಅಕ್ಕಿ ಲಾಭಾಂಶ ಹೆಚ್ಚಿಸಬಹುದು.",
    },
    "vegetative": {
        "en": "Split fertilizer doses and sell grain plus fodder or residue to improve total field income.",
        "kn": "ರಸಗೊಬ್ಬರವನ್ನು ಹಂತ ಹಂತವಾಗಿ ನೀಡಿ ಮತ್ತು ಧಾನ್ಯ ಜೊತೆಗೆ ಹೊಲ್ಲು ಅಥವಾ ಉಳಿದ ಅವಶೇಷಗಳನ್ನು ಮಾರಾಟ ಮಾಡಿ ಒಟ್ಟು ಆದಾಯವನ್ನು ಹೆಚ್ಚಿಸಿ.",
    },
}


def bilingual(en_text, kn_text):
    return {"en": en_text, "kn": kn_text}


def crop_view(crop_name):
    crop = CROP_LIBRARY[crop_name]
    return {"key": crop_name, "en": crop_name, "kn": crop["kn"]}


def district_view(name):
    district = DISTRICT_PROFILES[name]
    return {"en": name, "kn": district["kn"]}


def round_float(value, digits=1):
    return round(float(value), digits)


def load_soil_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        from model_training import train_and_save

        return train_and_save()

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        return bundle
    return {"pipeline": bundle, "feature_columns": FEATURE_COLUMNS}


def get_dashboard_config():
    districts = []
    for name, profile in DISTRICT_PROFILES.items():
        districts.append(
            {
                "name": district_view(name),
                "avg_rainfall": profile["avg_rainfall"],
                "note": bilingual(
                    profile["note"],
                    f"{profile['kn']} ಜಿಲ್ಲೆಯಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ {profile['avg_rainfall']} ಮಿಮೀ ಮಳೆ ಲಭ್ಯವಾಗುತ್ತದೆ.",
                ),
            }
        )

    crops = [crop_view(name) for name in CROP_LIBRARY]
    return {
        "districts": districts,
        "crops": crops,
        "default_district": "Mandya",
        "default_language": "en",
    }


def default_rainfall(district):
    return float(DISTRICT_PROFILES.get(district, DISTRICT_PROFILES["Mandya"])["avg_rainfall"])


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_number(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def normalize_reading(reading, district):
    reading = dict(reading or {})
    moisture = reading.get("moisture", reading.get("humidity"))
    rainfall = reading.get("rainfall", default_rainfall(district))

    values = {
        "moisture": normalize_number(moisture, "moisture"),
        "ph": normalize_number(reading.get("ph"), "ph"),
        "ec": normalize_number(reading.get("ec"), "ec"),
        "temperature": normalize_number(reading.get("temperature"), "temperature"),
        "nitrogen": normalize_number(reading.get("nitrogen"), "nitrogen"),
        "phosphorus": normalize_number(reading.get("phosphorus"), "phosphorus"),
        "potassium": normalize_number(reading.get("potassium"), "potassium"),
        "rainfall": normalize_number(rainfall, "rainfall"),
    }

    return {
        "moisture": round_float(clamp(values["moisture"], 0, 100), 1),
        "ph": round_float(clamp(values["ph"], 3.0, 10.0), 2),
        "ec": round_float(clamp(values["ec"], 0.0, 5.0), 2),
        "temperature": round_float(clamp(values["temperature"], 0.0, 50.0), 1),
        "nitrogen": round_float(clamp(values["nitrogen"], 0.0, 250.0), 1),
        "phosphorus": round_float(clamp(values["phosphorus"], 0.0, 200.0), 1),
        "potassium": round_float(clamp(values["potassium"], 0.0, 250.0), 1),
        "rainfall": round_float(clamp(values["rainfall"], 100.0, 2000.0), 1),
    }


def extract_samples(payload, district):
    samples = payload.get("samples")
    if isinstance(samples, list) and samples:
        return [normalize_reading(sample, district) for sample in samples[:6]]
    return [normalize_reading(payload, district)]


def average_samples(samples):
    return {
        feature: round_float(sum(sample[feature] for sample in samples) / len(samples), 2)
        for feature in THRESHOLDS
    }


def classify_status(value, feature_name):
    low, high = THRESHOLDS[feature_name]
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Medium"


def classify_statuses(values):
    return {feature: classify_status(values[feature], feature) for feature in THRESHOLDS}


def feature_label(feature, language="en"):
    return FEATURE_LABELS[feature][language]


def parameter_score(feature, value):
    low, high = THRESHOLDS[feature]
    midpoint = (low + high) / 2
    window = max((high - low) / 2, 1)

    if low <= value <= high:
        penalty = abs(value - midpoint) / window * 12
        return max(82, 100 - penalty)

    if value < low:
        distance = (low - value) / max(low, 1)
    else:
        distance = (value - high) / max(high, 1)

    return max(20, 74 - distance * 115)


def soil_health_score(values):
    weights = {
        "moisture": 1.1,
        "ph": 1.2,
        "ec": 1.0,
        "temperature": 0.8,
        "nitrogen": 1.3,
        "phosphorus": 1.1,
        "potassium": 1.1,
        "rainfall": 0.8,
    }
    weighted_total = sum(parameter_score(feature, values[feature]) * weight for feature, weight in weights.items())
    weight_sum = sum(weights.values())
    return round(weighted_total / weight_sum)


def soil_health_status(score):
    if score >= 78:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def main_issue(statuses):
    for feature in ISSUE_PRIORITY:
        if statuses[feature] != "Medium":
            return {"feature": feature, "status": statuses[feature]}
    return {"feature": "nitrogen", "status": "Medium"}


def main_issue_text(issue):
    status = issue["status"]
    feature = issue["feature"]
    if status == "Medium":
        return bilingual(
            "Most core parameters are in the target range.",
            "ಹೆಚ್ಚಿನ ಪ್ರಮುಖ ಪರಿಮಾಣಗಳು ಗುರಿ ಮಿತಿಯೊಳಗಿವೆ.",
        )
    return bilingual(
        f"{feature_label(feature, 'en')} is {status.lower()} and needs attention.",
        f"{feature_label(feature, 'kn')} {STATUS_TRANSLATIONS[status]} ಮಟ್ಟದಲ್ಲಿದ್ದು ಗಮನ ಅಗತ್ಯವಿದೆ.",
    )


def model_frame(values, district):
    record = dict(values)
    record["district"] = district
    return pd.DataFrame([record], columns=FEATURE_COLUMNS)


def ai_crop_recommendation(values, district, model_bundle):
    pipeline = model_bundle["pipeline"]
    frame = model_frame(values, district)
    predicted = pipeline.predict(frame)[0]

    confidence = None
    alternatives = []
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        confidence = round_float(ranked[0][1], 2)
        alternatives = [label for label, _ in ranked[1:3]]

    selected = predicted if predicted in CROP_LIBRARY else "Maize"
    filtered_alternatives = [label for label in alternatives if label in CROP_LIBRARY and label != selected]

    return {
        "selected": selected,
        "source": "ai",
        "confidence": confidence,
        "alternatives": filtered_alternatives,
    }


def resolve_crop(payload, values, district, model_bundle, alerts):
    operation_mode = str(payload.get("operation_mode", "ai")).lower()
    manual_crop = payload.get("crop")
    ai_result = ai_crop_recommendation(values, district, model_bundle)

    if operation_mode == "manual":
        if manual_crop in CROP_LIBRARY:
            alternatives = [name for name in ai_result["alternatives"] if name != manual_crop]
            return {
                "selected": manual_crop,
                "source": "manual",
                "confidence": ai_result["confidence"],
                "alternatives": alternatives[:2],
                "ai_reference": ai_result["selected"],
            }

        alerts.append(
            {
                "level": "warning",
                "message": bilingual(
                    "Manual mode was selected without a valid crop, so the AI recommendation was used.",
                    "ಮಾನ್ಯ ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡದ ಕಾರಣ ಕೈಚಾಲಿತ ಮೋಡ್‌ನಲ್ಲಿ AI ಶಿಫಾರಸನ್ನು ಬಳಸಲಾಗಿದೆ.",
                ),
            }
        )

    ai_result["ai_reference"] = ai_result["selected"]
    return ai_result


def fertilizer_recommendation(crop_name, nutrient_statuses, planting_mode):
    crop = CROP_LIBRARY[crop_name]
    nutrient_plan = {}
    nutrient_gaps = {}

    for nutrient, required in crop["requirements"].items():
        status = nutrient_statuses[nutrient]
        available = required * STATUS_FACTORS[status]
        gap = max(required - available, 0)
        nutrient_gaps[nutrient] = round_float(gap, 1)
        nutrient_plan[nutrient] = {
            "status": status,
            "required": required,
            "available": round_float(available, 1),
            "gap": round_float(gap, 1),
        }

    fertilizers = {
        "urea": round_float(nutrient_gaps["nitrogen"] / 0.46, 1),
        "dap": round_float(nutrient_gaps["phosphorus"] / 0.46, 1),
        "mop": round_float(nutrient_gaps["potassium"] / 0.60, 1),
    }

    if planting_mode == "pre":
        note = bilingual(
            "Pre-crop plan: apply the full recommendation during land preparation, then split nitrogen after emergence.",
            "ಬೆಳೆ ಮೊದಲು: ಭೂಮಿ ಸಿದ್ಧಪಡಿಸುವಾಗ ಪೂರ್ಣ ಶಿಫಾರಸನ್ನು ನೀಡಿ, ನಂತರ ಮೊಳೆತ ನಂತರ ನೈಟ್ರೋಜನ್ ಅನ್ನು ಹಂತ ಹಂತವಾಗಿ ನೀಡಿ.",
        )
    else:
        note = bilingual(
            "Post-crop correction: apply the recommendation as a corrective top-dress in split doses, not as one heavy application.",
            "ಬೆಳೆ ನಂತರದ ತಿದ್ದುಪಡಿ: ಈ ಪ್ರಮಾಣವನ್ನು ಒಂದು ಬಾರಿ ಹೆಚ್ಚು ಕೊಡದೆ, ಹಂತ ಹಂತವಾಗಿ ಮೇಲ್ಭಾಗದ ಗೊಬ್ಬರವಾಗಿ ನೀಡಿ.",
        )

    return {
        "crop": crop_view(crop_name),
        "nutrients": nutrient_plan,
        "fertilizers_kg_ha": fertilizers,
        "note": note,
    }


def irrigation_advice(values, district):
    if values["moisture"] < 25 or values["rainfall"] < default_rainfall(district) * 0.7:
        return {"code": "urgent", "message": IRRIGATION_TEMPLATES["urgent"]}
    if values["moisture"] > 70 or values["rainfall"] > default_rainfall(district) * 1.25:
        return {"code": "drainage", "message": IRRIGATION_TEMPLATES["drainage"]}
    return {"code": "balanced", "message": IRRIGATION_TEMPLATES["balanced"]}


def deficiency_detection(statuses):
    issues = []
    for nutrient in ("nitrogen", "phosphorus", "potassium"):
        if statuses[nutrient] == "Low":
            issues.append({"nutrient": feature_label(nutrient, "en"), "message": DEFICIENCY_LIBRARY[nutrient]})

    if not issues:
        issues.append(
            {
                "nutrient": "Balanced",
                "message": bilingual(
                    "No major NPK deficiency is detected from the current reading.",
                    "ಪ್ರಸ್ತುತ ಓದುಗಳಲ್ಲಿ ಪ್ರಮುಖ NPK ಕೊರತೆ ಕಂಡುಬಂದಿಲ್ಲ.",
                ),
            }
        )

    return issues


def weather_note(values, district):
    normal = default_rainfall(district)
    if values["rainfall"] < normal * 0.8:
        return bilingual(
            f"Rainfall is below the normal level for {district}. Prefer moisture-saving irrigation and lower-risk crop choices.",
            f"{DISTRICT_PROFILES[district]['kn']} ಜಿಲ್ಲೆಗೆ ಸಾಮಾನ್ಯ ಮಟ್ಟಕ್ಕಿಂತ ಮಳೆ ಕಡಿಮೆ ಇದೆ. ತೇವಾಂಶ ಉಳಿಸುವ ನೀರಾವರಿ ಮತ್ತು ಕಡಿಮೆ ಅಪಾಯದ ಬೆಳೆಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ.",
        )
    if values["rainfall"] > normal * 1.2:
        return bilingual(
            f"Rainfall is above the usual level for {district}. Keep drainage ready and watch salinity and root stress.",
            f"{DISTRICT_PROFILES[district]['kn']} ಜಿಲ್ಲೆಗೆ ಸಾಮಾನ್ಯ ಮಟ್ಟಕ್ಕಿಂತ ಮಳೆ ಹೆಚ್ಚು ಇದೆ. ನೀರು ಹರಿಸುವ ವ್ಯವಸ್ಥೆಯನ್ನು ಸಿದ್ಧವಾಗಿಡಿ ಮತ್ತು ಬೇರು ಒತ್ತಡ ಹಾಗೂ ಉಪ್ಪುದ್ರವ್ಯವನ್ನು ಗಮನಿಸಿ.",
        )
    return bilingual(
        f"Rainfall is close to the normal level for {district}. Use district-standard irrigation scheduling.",
        f"{DISTRICT_PROFILES[district]['kn']} ಜಿಲ್ಲೆಗೆ ಮಳೆಯ ಪ್ರಮಾಣ ಸಾಮಾನ್ಯ ಮಟ್ಟಕ್ಕೆ ಸಮೀಪದಲ್ಲಿದೆ. ಜಿಲ್ಲೆಯ ಸಾಮಾನ್ಯ ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿಯನ್ನು ಅನುಸರಿಸಿ.",
    )


def crop_rotation_suggestion(crop_name):
    rotation = CROP_LIBRARY[crop_name]["rotation"]
    return bilingual(
        rotation,
        f"ಬೆಳೆ ಚಕ್ರ ಸಲಹೆ: {crop_view(crop_name)['kn']} ನಂತರ ಮಣ್ಣಿನ ಸಮತೋಲನಕ್ಕಾಗಿ ಕಾಳು ಬೆಳೆ ಅಥವಾ ಪರ್ಯಾಯ ಬೆಳೆ ಬೆಳೆಸಿ.",
    )


def income_suggestion(crop_name, soil_status):
    if soil_status == "Low":
        return INCOME_TEMPLATES["low_soil"]
    if crop_name == "Rice":
        return INCOME_TEMPLATES["rice"]
    return bilingual(CROP_LIBRARY[crop_name]["income"], f"ಆದಾಯ ಸಲಹೆ: {crop_view(crop_name)['kn']} ಬೆಳೆಗಾಗಿ ಹಂತ ಹಂತದ ಗೊಬ್ಬರ ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಯೋಜನೆ ಬಳಸಿ.")


def build_alerts(values, statuses, district, sample_mode, sample_count):
    alerts = []

    if statuses["moisture"] == "Low":
        alerts.append({"level": "critical", "message": bilingual("Critical moisture deficit detected.", "ಗಂಭೀರ ತೇವಾಂಶ ಕೊರತೆ ಕಂಡುಬಂದಿದೆ.")})
    if statuses["ec"] == "High":
        alerts.append({"level": "critical", "message": bilingual("Soil EC is high. Salinity management is required.", "ಮಣ್ಣಿನ ಇಸಿ ಹೆಚ್ಚಿದೆ. ಉಪ್ಪುದ್ರವ್ಯ ನಿಯಂತ್ರಣ ಅಗತ್ಯವಿದೆ.")})
    if statuses["ph"] == "Low":
        alerts.append({"level": "warning", "message": bilingual("Soil pH is acidic. Lime or organic correction may be needed.", "ಮಣ್ಣಿನ ಪಿಎಚ್ ಆಮ್ಲೀಯವಾಗಿದೆ. ಚೂನಾ ಅಥವಾ ಸಾವಯವ ತಿದ್ದುಪಡಿ ಬೇಕಾಗಬಹುದು.")})
    if statuses["ph"] == "High":
        alerts.append({"level": "warning", "message": bilingual("Soil pH is alkaline. Use organic matter and avoid excess salts.", "ಮಣ್ಣಿನ ಪಿಎಚ್ ಕ್ಷಾರೀಯವಾಗಿದೆ. ಸಾವಯವ ಪದಾರ್ಥ ಬಳಸಿ ಮತ್ತು ಹೆಚ್ಚು ಉಪ್ಪುಗಳಿಂದ ದೂರವಿರಿ.")})
    if values["rainfall"] < default_rainfall(district) * 0.75:
        alerts.append({"level": "warning", "message": bilingual("Rainfall is below the district normal.", "ಜಿಲ್ಲೆಯ ಸಾಮಾನ್ಯ ಮಟ್ಟಕ್ಕಿಂತ ಮಳೆಯ ಪ್ರಮಾಣ ಕಡಿಮೆಯಾಗಿದೆ.")})
    if sample_mode == "multiple" and sample_count < 4:
        alerts.append({"level": "info", "message": bilingual("Add 4 to 6 zig-zag samples for stronger large-land analysis.", "ದೊಡ್ಡ ಜಮೀನು ವಿಶ್ಲೇಷಣೆಗೆ 4 ರಿಂದ 6 ಜಿಗ್-ಜಾಗ್ ಮಾದರಿಗಳನ್ನು ಸೇರಿಸಿ.")})

    if not alerts:
        alerts.append({"level": "info", "message": bilingual("No critical field alerts right now.", "ಪ್ರಸ್ತುತ ಯಾವುದೇ ಗಂಭೀರ ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ.")})

    return alerts


def explanation_points(values, district, crop_result, issue, fertilizer):
    biggest_gap = max(fertilizer["nutrients"].items(), key=lambda item: item[1]["gap"])
    nutrient_name = feature_label(biggest_gap[0], "en")
    nutrient_name_kn = feature_label(biggest_gap[0], "kn")

    source_text = "AI model selected the crop" if crop_result["source"] == "ai" else "Farmer selected the crop manually"
    source_text_kn = "AI ಮಾದರಿ ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿದೆ" if crop_result["source"] == "ai" else "ರೈತರು ಬೆಳೆ ಕೈಯಾರೆ ಆಯ್ಕೆ ಮಾಡಿದ್ದಾರೆ"

    return [
        bilingual(
            f"District context: {district} averages about {int(default_rainfall(district))} mm rainfall. {DISTRICT_PROFILES[district]['note']}",
            f"ಜಿಲ್ಲಾ ಹಿನ್ನೆಲೆ: {DISTRICT_PROFILES[district]['kn']} ಜಿಲ್ಲೆಯಲ್ಲಿ ಸರಾಸರಿ {int(default_rainfall(district))} ಮಿಮೀ ಮಳೆ ಬೀಳುತ್ತದೆ.",
        ),
        bilingual(
            f"Crop decision: {source_text} and recommended {crop_result['selected']}.",
            f"ಬೆಳೆ ನಿರ್ಧಾರ: {source_text_kn} ಮತ್ತು {crop_view(crop_result['selected'])['kn']} ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ.",
        ),
        main_issue_text(issue),
        bilingual(
            f"Fertilizer logic: {nutrient_name} has the biggest gap, so its correction drives the kg/ha recommendation.",
            f"ರಸಗೊಬ್ಬರ ಲಾಜಿಕ್: {nutrient_name_kn} ಅತಿದೊಡ್ಡ ಕೊರತೆ ಹೊಂದಿರುವುದರಿಂದ kg/ha ಶಿಫಾರಸಿನಲ್ಲಿ ಅದೇ ಮುಖ್ಯವಾಗಿದೆ.",
        ),
    ]


def zone_analysis(samples, district, model_bundle):
    zones = []
    for index, sample in enumerate(samples, start=1):
        statuses = classify_statuses(sample)
        score = soil_health_score(sample)
        health = soil_health_status(score)
        issue = main_issue(statuses)
        crop = ai_crop_recommendation(sample, district, model_bundle)
        zones.append(
            {
                "zone": index,
                "soil_health_status": health,
                "soil_health_score": score,
                "crop": crop_view(crop["selected"]),
                "key_issue": main_issue_text(issue),
                "values": sample,
            }
        )
    return zones


def selection_summary(language, district, planting_mode, operation_mode, data_mode, sample_mode, crop_result):
    crop_name = crop_result["selected"]
    if language == "kn":
        return (
            f"ಭಾಷೆ: ಕನ್ನಡ | ಜಿಲ್ಲೆ: {DISTRICT_PROFILES[district]['kn']} | ಹಂತ: {'ಬೆಳೆ ಮೊದಲು' if planting_mode == 'pre' else 'ಬೆಳೆ ನಂತರ'} | ಮೋಡ್: {'AI' if operation_mode == 'ai' else 'ಕೈಚಾಲಿತ'} | ಡೇಟಾ: {'ಲೈವ್' if data_mode == 'live' else 'ಡೆಮೊ'} | ಮಾದರಿ: {'ಏಕ' if sample_mode == 'single' else 'ಬಹು'} | ಬೆಳೆ: {crop_view(crop_name)['kn']}"
        )
    return (
        f"Language: English | District: {district} | Stage: {'Pre-crop' if planting_mode == 'pre' else 'Post-crop'} | Mode: {'AI' if operation_mode == 'ai' else 'Manual'} | Data: {'Live' if data_mode == 'live' else 'Demo'} | Sample: {'Single' if sample_mode == 'single' else 'Multiple'} | Crop: {crop_name}"
    )


def predict_soil(payload, model_bundle=None):
    model_bundle = model_bundle or load_soil_model()
    payload = dict(payload or {})

    district = payload.get("district") if payload.get("district") in DISTRICT_PROFILES else "Mandya"
    language = str(payload.get("language", "en")).lower()
    planting_mode = str(payload.get("planting_mode", "pre")).lower()
    operation_mode = str(payload.get("operation_mode", "ai")).lower()
    data_mode = str(payload.get("data_mode", "demo")).lower()
    sample_mode = str(payload.get("sample_mode", "single")).lower()
    analysis_type = str(payload.get("analysis_type", "average")).lower()

    samples = extract_samples(payload, district)
    average_values = average_samples(samples)
    statuses = classify_statuses(average_values)
    score = soil_health_score(average_values)
    health = soil_health_status(score)
    issue = main_issue(statuses)

    alerts = []
    crop_result = resolve_crop(payload, average_values, district, model_bundle, alerts)
    fertilizer = fertilizer_recommendation(crop_result["selected"], statuses, planting_mode)
    irrigation = irrigation_advice(average_values, district)
    deficiencies = deficiency_detection(statuses)
    alerts.extend(build_alerts(average_values, statuses, district, sample_mode, len(samples)))
    weather = weather_note(average_values, district)
    rotation = crop_rotation_suggestion(crop_result["selected"])
    income = income_suggestion(crop_result["selected"], health)
    explanations = explanation_points(average_values, district, crop_result, issue, fertilizer)

    summary_en = selection_summary("en", district, planting_mode, operation_mode, data_mode, sample_mode, crop_result)
    summary_kn = selection_summary("kn", district, planting_mode, operation_mode, data_mode, sample_mode, crop_result)

    result = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "language": language,
        "district": district_view(district),
        "district_key": district,
        "planting_mode": planting_mode,
        "operation_mode": operation_mode,
        "data_mode": data_mode,
        "sample_mode": sample_mode,
        "analysis_type": analysis_type,
        "sample_count": len(samples),
        "average_values": average_values,
        "statuses": statuses,
        "soil_health": {
            "status": health,
            "status_kn": STATUS_TRANSLATIONS[health],
            "score": score,
            "summary": bilingual(
                f"Soil health is {health} with a score of {score}/100.",
                f"ಮಣ್ಣಿನ ಆರೋಗ್ಯ {STATUS_TRANSLATIONS[health]} ಮಟ್ಟದಲ್ಲಿದೆ ಮತ್ತು ಅಂಕ {score}/100 ಆಗಿದೆ.",
            ),
            "key_issue": main_issue_text(issue),
        },
        "crop_recommendation": {
            "selected": crop_view(crop_result["selected"]),
            "source": crop_result["source"],
            "confidence": crop_result["confidence"],
            "alternatives": [crop_view(name) for name in crop_result.get("alternatives", [])],
            "ai_reference": crop_view(crop_result["ai_reference"]),
        },
        "fertilizer_recommendation": fertilizer,
        "irrigation_advice": irrigation,
        "deficiency_detection": deficiencies,
        "weather_note": weather,
        "crop_rotation_suggestion": rotation,
        "income_improvement_suggestion": income,
        "alerts": alerts,
        "explanations": explanations,
        "selection_summary": bilingual(summary_en, summary_kn),
        "farmer_message": bilingual(
            f"Recommended crop: {crop_result['selected']}. Apply Urea {fertilizer['fertilizers_kg_ha']['urea']} kg/ha, DAP {fertilizer['fertilizers_kg_ha']['dap']} kg/ha, and MOP {fertilizer['fertilizers_kg_ha']['mop']} kg/ha.",
            f"ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ: {crop_view(crop_result['selected'])['kn']}. ಯೂರಿಯಾ {fertilizer['fertilizers_kg_ha']['urea']} kg/ha, ಡಿಎಪಿ {fertilizer['fertilizers_kg_ha']['dap']} kg/ha ಮತ್ತು ಎಂಒಪಿ {fertilizer['fertilizers_kg_ha']['mop']} kg/ha ನೀಡಿ.",
        ),
    }

    if sample_mode == "multiple" and analysis_type == "zone":
        result["zones"] = zone_analysis(samples, district, model_bundle)
    else:
        result["zones"] = []

    return result
