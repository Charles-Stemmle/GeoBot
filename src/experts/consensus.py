# 1. IMPORTS

from pathlib import Path
import numpy as np
import tensorflow as tf


# 2. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MODEL_FOLDER = PROJECT_FOLDER / "models"

IMAGE_SIZE = (224, 224)

# Initial weights.
# These are starting values, not optimized weights.
MODEL_WEIGHTS = {
    "holistic": 0.50,
    "road": 0.25,
    "environment": 0.25,
}

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


# 3. LOAD MODELS AND COUNTRY LABELS

models = {}
country_labels = {}

for model_name, (model_file, labels_file) in MODEL_FILES.items():
    model_path = MODEL_FOLDER / model_file
    labels_path = MODEL_FOLDER / labels_file

    models[model_name] = tf.keras.models.load_model(model_path)

    with open(labels_path, "r", encoding="utf-8") as file:
        country_labels[model_name] = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(f"Loaded {model_name} model")
    print(f"  Countries: {len(country_labels[model_name])}")


# 4. LOAD AND PREPARE AN IMAGE

def load_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


# 5. GET EACH MODEL'S PREDICTIONS

def predict(image_path):
    image_array = load_image(image_path)

    predictions = {}

    for model_name, model in models.items():
        probabilities = model.predict(
            image_array,
            verbose=0,
        )[0]

        labels = country_labels[model_name]

        if len(probabilities) != len(labels):
            raise ValueError(
                f"{model_name}: model returned "
                f"{len(probabilities)} predictions, but "
                f"{len(labels)} country labels were loaded."
            )

        predictions[model_name] = dict(
            zip(labels, probabilities)
        )

    return predictions


# 6. COMBINE MODEL PREDICTIONS

def get_consensus(predictions):
    combined_scores = {}

    for model_name, country_scores in predictions.items():
        weight = MODEL_WEIGHTS[model_name]

        for country, probability in country_scores.items():
            combined_scores[country] = (
                combined_scores.get(country, 0.0)
                + weight * float(probability)
            )

    total = sum(combined_scores.values())

    if total <= 0:
        raise ValueError("Consensus scores have no positive total.")

    # Normalize the combined scores so they sum to 1.
    combined_scores = {
        country: score / total
        for country, score in combined_scores.items()
    }

    return sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# 7. DISPLAY PREDICTIONS

def display_predictions(image_path):
    predictions = predict(image_path)
    consensus = get_consensus(predictions)

    print("\n" + "=" * 45)
    print(f"IMAGE: {Path(image_path).name}")
    print("=" * 45)

    for model_name, country_scores in predictions.items():
        print(f"\n{model_name.upper()} TOP 5")

        top_five = sorted(
            country_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        for rank, (country, probability) in enumerate(
            top_five, start=1
        ):
            print(f"{rank}. {country}: {probability:.2%}")

    print("\nFINAL CONSENSUS TOP 5")

    for rank, (country, probability) in enumerate(
        consensus[:5], start=1
    ):
        print(f"{rank}. {country}: {probability:.2%}")


# 8. RUN A TEST

if __name__ == "__main__":
    image_path = input