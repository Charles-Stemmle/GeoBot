import tensorflow as tf
import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_FOLDER / "models" / "geobot_country_classifier.keras"
LABELS_PATH = PROJECT_FOLDER / "models" / "country_labels.txt"
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "prediction"

# Change path to image being tested
IMAGE_PATH = PROJECT_FOLDER / "data" / "photos" / "dataset" / "prediction" / "bigtone.png"
IMAGE_SIZE = (224, 224)


print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

#country labels
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    countries = [line.strip() for line in file if line.strip()]

print("Loading image:", IMAGE_PATH)

image = tf.io.read_file(str(IMAGE_PATH))
image = tf.io.decode_png(image, channels=3)
image = tf.image.resize(image, IMAGE_SIZE)

image = tf.keras.applications.mobilenet_v2.preprocess_input(
    image
)

image = tf.expand_dims(image, axis=0)  # Add batch dimension

print("Predicting country...")

predictions = model.predict(image, verbose=0)[0]


# -----------------------------
# 7. DISPLAY TOP FIVE COUNTRIES
# -----------------------------

top_indices = predictions.argsort()[-5:][::-1]

print("\nTop 5 predictions:")

for rank, index in enumerate(top_indices, start=1):
    country = countries[index]
    probability = predictions[index] * 100

    print(f"{rank}. {country}: {probability:.2f}%")


#print best guesses

best_index = predictions.argmax()
best_country = countries[best_index]

print("\nPredicted country:", best_country)