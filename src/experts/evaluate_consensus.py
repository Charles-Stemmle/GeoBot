# 1. IMPORTS

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# 2. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
MODEL_FOLDER = PROJECT_FOLDER / "models"

DATASET_FILE = PROJECT_FOLDER / "src" / "combined_dataset.csv"
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "dataset"
ROAD_FOLDER = PROJECT_FOLDER / "data" / "photos" / "road"
ENVIRONMENT_FOLDER = PROJECT_FOLDER / "data" / "photos" / "environment"
RESULTS_FOLDER = PROJECT_FOLDER / "data" / "analysis"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
MAX_TEST_IMAGES = 500

MODEL_WEIGHTS = {
    "holistic": 0.50,
    "road": 0.25,
    "environment": 0.25,
}


# 3. LOAD DATASET

print("Loading dataset...")

data = pd.read_csv(DATASET_FILE)

required_columns = {"filename", "country"}

if not required_columns.issubset(data.columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}"
    )

data = data.dropna(subset=["filename", "country"]).copy()
data["filename"] = data["filename"].astype(str)
data["country"] = data["country"].astype(str)

print(f"Dataset rows: {len(data)}")


# 4. LOAD MODELS AND COUNTRY LABELS

MODEL_FILES = {
    "holistic": (
        "geobot_country_classifier.keras",
        "country_labels.txt",
    ),
    "road": (
        "geobot_road_classifier.keras",
        "road_country_labels.txt",
    ),
    "environment": (
        "geobot_environment_classifier.keras",
        "environment_country_labels.txt",
    ),
}

models = {}
country_labels = {}

for model_name, (model_file, labels_file) in MODEL_FILES.items():
    print(f"Loading {model_name} model...")

    models[model_name] = tf.keras.models.load_model(
        MODEL_FOLDER / model_file
    )

    with open(
        MODEL_FOLDER / labels_file,
        "r",
        encoding="utf-8",
    ) as file:
        country_labels[model_name] = [
            line.strip()
            for line in file
            if line.strip()
        ]

    output_count = models[model_name].output_shape[-1]

    if output_count != len(country_labels[model_name]):
        raise ValueError(
            f"{model_name}: output count does not match label count."
        )

print("All models loaded.")


# 5. FIND VALID IMAGES

def get_image_path(filename, folder):
    # Dataset filenames might use a different extension
    # from the saved segmentation masks.
    return folder / filename


valid_rows = []

for _, row in data.iterrows():
    filename = Path(row["filename"]).name

    original_path = get_image_path(filename, IMAGE_FOLDER)
    road_path = ROAD_FOLDER / f"{Path(filename).stem}.png"
    environment_path = (
        ENVIRONMENT_FOLDER / f"{Path(filename).stem}.png"
    )

    if (
        original_path.is_file()
        and road_path.is_file()
        and environment_path.is_file()
    ):
        valid_rows.append({
            "filename": filename,
            "country": row["country"],
        })

valid_data = pd.DataFrame(valid_rows).drop_duplicates(
    subset="filename"
).reset_index(drop=True)

if valid_data.empty:
    raise ValueError(
        "No images have all three required files. "
        "Check the dataset and segmentation folders."
    )

# This is a reproducible sample, not necessarily a held-out test set.
if len(valid_data) > MAX_TEST_IMAGES:
    valid_data = valid_data.sample(
        n=MAX_TEST_IMAGES,
        random_state=42,
    ).reset_index(drop=True)

print(f"Images selected for evaluation: {len(valid_data)}")


# 6. IMAGE PREPROCESSING

def load_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


# 7. GET TOP PREDICTIONS

def get_top_predictions(model_name, image_path):
    image_array = load_image(image_path)

    probabilities = models[model_name].predict(
        image_array,
        verbose=0,
    )[0]

    labels = country_labels[model_name]

    return {
        labels[index]: float(probabilities[index])
        for index in range(len(labels))
    }


# 8. COMBINE PREDICTIONS

def get_consensus(predictions):
    scores = {}

    for model_name, country_scores in predictions.items():
        weight = MODEL_WEIGHTS[model_name]

        for country, probability in country_scores.items():
            scores[country] = (
                scores.get(country, 0.0)
                + weight * probability
            )

    total = sum(scores.values())

    if total <= 0:
        raise ValueError("Consensus scores have no positive total.")

    return {
        country: score / total
        for country, score in scores.items()
    }


# 9. EVALUATE IMAGES

results = []

for index, row in valid_data.iterrows():
    filename = row["filename"]
    true_country = row["country"]

    original_path = IMAGE_FOLDER / filename
    road_path = ROAD_FOLDER / f"{Path(filename).stem}.png"
    environment_path = (
        ENVIRONMENT_FOLDER / f"{Path(filename).stem}.png"
    )

    try:
        predictions = {
            "holistic": get_top_predictions(
                "holistic", original_path
            ),
            "road": get_top_predictions(
                "road", road_path
            ),
            "environment": get_top_predictions(
                "environment", environment_path
            ),
        }

        predictions["consensus"] = get_consensus(
            predictions
        )

        result = {
            "filename": filename,
            "true_country": true_country,
        }

        for model_name, country_scores in predictions.items():
            ranked = sorted(
                country_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_five = [country for country, _ in ranked[:5]]

            result[f"{model_name}_top1"] = (
                ranked[0][0] if ranked else ""
            )
            result[f"{model_name}_top5"] = ", ".join(top_five)
            result[f"{model_name}_top1_correct"] = (
                bool(ranked and ranked[0][0] == true_country)
            ) 
            result[f"{model_name}_top5_correct"] = (
                true_country in top_five
            )

        results.append(result)

        if (index + 1) % 25 == 0 or index + 1 == len(valid_data):
            print(
                f"Evaluated {index + 1}/{len(valid_data)} images"
            )

    except Exception as error:
        print(f"Failed on {filename}: {error}")


# 10. SAVE RESULTS

if not results:
    raise ValueError("No images were evaluated successfully.")

results_df = pd.DataFrame(results)

results_path = RESULTS_FOLDER / "consensus_evaluation_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\nPer-image results saved to: {results_path}")


# 11. DISPLAY ACCURACY

model_names = [
    "holistic",
    "road",
    "environment",
    "consensus",
]

summary = []

for model_name in model_names:
    summary.append({
        "model": model_name,
        "top1_accuracy": (
            results_df[f"{model_name}_top1_correct"].mean() * 100
        ),
        "top5_accuracy": (
            results_df[f"{model_name}_top5_correct"].mean() * 100
        ),
        "images_evaluated": len(results_df),
    })

summary_df = pd.DataFrame(summary)

print("\nEVALUATION SUMMARY")
print(summary_df.to_string(index=False, float_format="%.2f"))

summary_path = RESULTS_FOLDER / "consensus_evaluation_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nSummary saved to: {summary_path}")