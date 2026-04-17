from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "soil_data.csv"
SENSOR_DATA_PATH = BASE_DIR / "data" / "soil_.csv"
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
TARGET_COLUMN = "crop_label"

DISTRICT_PROFILES = {
    "Mandya": {"avg_rainfall": 720},
    "Mysore": {"avg_rainfall": 760},
    "Davangere": {"avg_rainfall": 640},
    "Belagavi": {"avg_rainfall": 810},
    "Shivamogga": {"avg_rainfall": 1280},
    "Tumakuru": {"avg_rainfall": 690},
    "Raichur": {"avg_rainfall": 590},
    "Vijayapura": {"avg_rainfall": 560},
    "Kolar": {"avg_rainfall": 700},
    "Hassan": {"avg_rainfall": 1040},
}

CROP_LABEL_MAP = {
    "Paddy": "Rice",
    "Maize": "Maize",
    "Wheat": "Wheat",
    "Millets": "Millets",
    "Pulses": "Pulses",
    "Sugarcane": "Sugarcane",
    "Ground Nuts": "Groundnut",
    "Cotton": "Cotton",
    "Oil seeds": "Oil Seeds",
    "Tobacco": "Tobacco",
    "Barley": "Barley",
}

CROP_DISTRICT_MAP = {
    "Rice": ["Mandya", "Mysore", "Raichur", "Shivamogga"],
    "Sugarcane": ["Mandya", "Belagavi", "Mysore", "Hassan"],
    "Maize": ["Davangere", "Tumakuru", "Hassan", "Belagavi"],
    "Wheat": ["Belagavi", "Vijayapura", "Raichur", "Davangere"],
    "Millets": ["Tumakuru", "Davangere", "Kolar", "Vijayapura"],
    "Pulses": ["Raichur", "Kolar", "Davangere", "Tumakuru"],
    "Groundnut": ["Kolar", "Tumakuru", "Raichur", "Davangere"],
    "Cotton": ["Raichur", "Davangere", "Belagavi", "Vijayapura"],
    "Oil Seeds": ["Kolar", "Tumakuru", "Davangere", "Belagavi"],
    "Tobacco": ["Mysore", "Mandya", "Belagavi", "Hassan"],
    "Barley": ["Belagavi", "Vijayapura", "Raichur", "Davangere"],
}

SOIL_BASE_PH = {
    "Black": 7.5,
    "Clayey": 6.9,
    "Loamy": 6.8,
    "Red": 6.3,
    "Sandy": 6.6,
}

SOIL_BASE_EC = {
    "Black": 1.35,
    "Clayey": 1.12,
    "Loamy": 0.96,
    "Red": 0.78,
    "Sandy": 0.62,
}

CROP_RAINFALL_HINT = {
    "Rice": 1100,
    "Sugarcane": 1200,
    "Maize": 720,
    "Wheat": 620,
    "Millets": 540,
    "Pulses": 560,
    "Groundnut": 610,
    "Cotton": 680,
    "Oil Seeds": 580,
    "Tobacco": 650,
    "Barley": 600,
}


def clip(value, low, high):
    return round(max(low, min(high, value)), 2)


def load_raw_dataset(path=RAW_DATA_PATH):
    dataset = pd.read_csv(path)
    dataset.columns = [column.strip().lower().replace(" ", "_") for column in dataset.columns]
    dataset = dataset.rename(
        columns={
            "temparature": "temperature",
            "soil_type": "soil_type",
            "crop_type": "crop_type",
            "phosphorous": "phosphorus",
            "fertilizer_name": "fertilizer_name",
        }
    )
    return dataset


def assign_district(crop_label, index):
    districts = CROP_DISTRICT_MAP.get(crop_label, list(DISTRICT_PROFILES))
    return districts[index % len(districts)]


def enrich_primary_dataset():
    dataset = load_raw_dataset()
    dataset[TARGET_COLUMN] = dataset["crop_type"].map(CROP_LABEL_MAP).fillna(dataset["crop_type"])

    districts = []
    rainfalls = []
    ph_values = []
    ec_values = []

    for index, row in dataset.iterrows():
        crop_label = row[TARGET_COLUMN]
        district = assign_district(crop_label, index)
        rainfall_target = CROP_RAINFALL_HINT.get(crop_label, 700)
        district_rainfall = DISTRICT_PROFILES[district]["avg_rainfall"]

        ph_value = SOIL_BASE_PH.get(row["soil_type"], 6.8)
        ph_value += (float(row["moisture"]) - 45) / 180
        ph_value -= (float(row["phosphorus"]) - 24) / 420

        ec_value = SOIL_BASE_EC.get(row["soil_type"], 0.9)
        ec_value += float(row["potassium"]) / 260
        ec_value -= float(row["moisture"]) / 240
        ec_value += float(row["humidity"]) / 420

        seasonal_shift = ((index % 7) - 3) * 18
        rainfall = district_rainfall * 0.65 + rainfall_target * 0.35 + seasonal_shift

        districts.append(district)
        rainfalls.append(clip(rainfall, 350, 1600))
        ph_values.append(clip(ph_value, 5.2, 8.4))
        ec_values.append(clip(ec_value, 0.3, 2.8))

    dataset["district"] = districts
    dataset["rainfall"] = rainfalls
    dataset["ph"] = ph_values
    dataset["ec"] = ec_values

    return dataset[
        [
            "moisture",
            "ph",
            "ec",
            "temperature",
            "nitrogen",
            "phosphorus",
            "potassium",
            "rainfall",
            "district",
            TARGET_COLUMN,
        ]
    ]


def crop_from_sensor_row(row):
    if row["moisture"] >= 58 or row["rainfall"] >= 950:
        return "Rice"
    if row["moisture"] < 25 and row["nitrogen"] < 70:
        return "Millets"
    if row["nitrogen"] < 55:
        return "Pulses"
    if row["temperature"] <= 24:
        return "Wheat"
    if row["phosphorus"] >= 45 and row["potassium"] >= 60:
        return "Groundnut"
    return "Maize"


def enrich_sensor_dataset(path=SENSOR_DATA_PATH):
    sensor = pd.read_csv(path)
    sensor.columns = [column.strip().lower() for column in sensor.columns]

    districts = []
    rainfalls = []
    crop_labels = []
    district_names = list(DISTRICT_PROFILES)

    for index, row in sensor.iterrows():
        district = district_names[index % len(district_names)]
        rainfall = DISTRICT_PROFILES[district]["avg_rainfall"] * 0.6 + float(row["moisture"]) * 8

        sensor_row = {
            "moisture": float(row["moisture"]),
            "ph": float(row["ph"]),
            "ec": float(row["ec"]),
            "temperature": float(row["temperature"]),
            "nitrogen": float(row["nitrogen"]),
            "phosphorus": float(row["phosphorus"]),
            "potassium": float(row["potassium"]),
            "rainfall": clip(rainfall, 350, 1600),
        }

        crop_labels.append(crop_from_sensor_row(sensor_row))
        districts.append(district)
        rainfalls.append(sensor_row["rainfall"])

    sensor["district"] = districts
    sensor["rainfall"] = rainfalls
    sensor[TARGET_COLUMN] = crop_labels

    return sensor[FEATURE_COLUMNS + [TARGET_COLUMN]]


def build_training_frame():
    primary = enrich_primary_dataset()
    sensor_ready = enrich_sensor_dataset()
    dataset = pd.concat([primary, sensor_ready], ignore_index=True)
    dataset["district"] = dataset["district"].astype(str)
    return dataset


def build_pipeline():
    numeric_features = [feature for feature in FEATURE_COLUMNS if feature != "district"]
    categorical_features = ["district"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def train_and_save():
    dataset = build_training_frame().dropna(subset=[TARGET_COLUMN])
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "districts": list(DISTRICT_PROFILES),
        "crop_labels": sorted(y.unique().tolist()),
        "training_accuracy": accuracy,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Saved model: {MODEL_PATH}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model_bundle


if __name__ == "__main__":
    train_and_save()
