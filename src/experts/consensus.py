# 1. IMPORTS

from pathlib import Path

import numpy as np
import tensorflow as tf


# 2. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
MODEL_FOLDER = PROJECT_FOLDER / "models"

ROAD_FOLDER = PROJECT_FOLDER / "data" / "photos" / "road"
ENVIRONMENT_FOLDER = PROJECT_FOLDER / "data" / "photos" / "environment"

IMAGE_SIZE = (224, 224)

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


# 3. LOAD MODELS AND LABELS

models = {}
country_labels = {}

for model_name, (model_file, labels_file) in MODEL_FILES.items():
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

    print(f"Loaded {model_name} model")


# 4. FIND THE MASKED IMAGE

def find_masked_image(original_path, folder):
    masked_path = folder / f"{original_path.stem}.png"

    if not masked_path.is_file():
        raise FileNotFoundError(
            f"Masked image not found: {masked_path}\n"
            "Make sure the segmentation script has processed this image."
        )

    return masked_path


# 5. LOAD AND PREPARE AN IMAGE

def load_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


# 6. GET PREDICTIONS FROM EACH MODEL

def predict(image_path):
    original_path = Path(image_path)

    road_path = find_masked_image(
        original_path,
        ROAD_FOLDER,
    )

    environment_path = find_masked_image(
        original_path,
        ENVIRONMENT_FOLDER,
    )

    # Each model receives the type of image it was trained on.
    model_inputs = {
        "holistic": original_path,
        "road": road_path,
        "environment": environment_path,
    }

    predictions = {}

    for model_name, model in models.items():
        image_array = load_image(model_inputs[model_name])

        probabilities = model.predict(
            image_array,
            verbose=0,
        )[0]

        labels = country_labels[model_name]

        if len(probabilities) != len(labels):
            raise ValueError(
                f"{model_name}: model output has "
                f"{len(probabilities)} values, but "
                f"{len(labels)} labels were loaded."
            )

        predictions[model_name] = dict(
            zip(labels, probabilities)
        )

    return predictions


# 7. COMBINE PREDICTIONS

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

    combined_scores = {
        country: score / total
        for country, score in combined_scores.items()
    }

    return sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# 8. DISPLAY RESULTS

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
            top_five,
            start=1,
        ):
            print(f"{rank}. {country}: {probability:.2%}")

    print("\nFINAL CONSENSUS TOP 5")

    for rank, (country, probability) in enumerate(
        consensus[:5],
        start=1,
    ):
        print(f"{rank}. {country}: {probability:.2%}")


# 9. RUN A TEST

if __name__ == "__main__":
    image_path = input(
        "\nEnter the path to a test image: "
    ).strip().strip('"')

    if not Path(image_path).is_file():
        print("Error: Original image not found.")
    else:
        display_predictions(image_path)