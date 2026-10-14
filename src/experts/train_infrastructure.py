from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATASET_FILE = PROJECT_FOLDER / "src" / "combined_dataset.csv"
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "infrastructure"
MODEL_FOLDER = PROJECT_FOLDER / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
RANDOM_SEED = 42


# 2. LOAD DATASET

print("Loading dataset...")

data = pd.read_csv(DATASET_FILE)

print("Dataset columns:", data.columns.tolist())
print("First filenames in CSV:", data["filename"].head().tolist())
print("Image folder:", IMAGE_FOLDER)
print("Folder exists:", IMAGE_FOLDER.exists())

if IMAGE_FOLDER.exists():
    print("First files in folder:", [
        path.name for path in list(IMAGE_FOLDER.iterdir())[:5]
    ])

# Keep only rows whose infrastructure mask exists.
data = data[
    data["filename"].apply(
        lambda filename: (
            IMAGE_FOLDER / Path(filename).with_suffix(".png").name
        ).is_file()
    )
].copy()

print("Images with infrastructure masks:", len(data))
print("Number of countries:", data["country"].nunique())

if len(data) == 0:
    raise ValueError(
        "No infrastructure masks were found. "
        "Check IMAGE_FOLDER and the image filenames printed above."
    )


# 3. SPLIT TRAINING AND VALIDATION DATA

# Remove countries with fewer than two images so stratification can work.
country_counts = data["country"].value_counts()
valid_countries = country_counts[country_counts >= 2].index
data = data[data["country"].isin(valid_countries)].copy()

print("Images after filtering rare countries:", len(data))

train, validation = train_test_split(
    data,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=data["country"]
)

print("Training images:", len(train))
print("Validation images:", len(validation))


# 4. ENCODE COUNTRY LABELS

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train["country"])
y_validation = label_encoder.transform(validation["country"])

num_classes = len(label_encoder.classes_)

print("Number of classes:", num_classes)
print("Countries:", list(label_encoder.classes_))


# 5. LOAD AND PROCESS MASKED IMAGES

def load_image(filename, label):
    filename = tf.strings.regex_replace(
        filename, r"\.[^.]+$", ".png"
    )

    image_path = tf.strings.join([
        str(IMAGE_FOLDER) + "\\",
        filename
    ])

    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# 6. CREATE INPUT PIPELINE

def create_dataset(dataframe, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["filename"].values, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


print("Creating TensorFlow datasets...")

train_dataset = create_dataset(train, y_train, shuffle=True)
validation_dataset = create_dataset(validation, y_validation)


# 7. BUILD MODEL

print("Building MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# 8. CONFIGURE TRAINING

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 9. TRAIN

print("\nStarting infrastructure specialist training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# 10. EVALUATE

print("\nEvaluating infrastructure specialist...")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
)

print(f"Validation loss: {validation_loss:.4f}")
print(f"Validation accuracy: {validation_accuracy:.2%}")


# 11. SAVE MODEL AND LABELS

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

model_path = MODEL_FOLDER / "geobot_infrastructure_classifier.keras"
model.save(model_path)

labels_path = MODEL_FOLDER / "infrastructure_country_labels.txt"

with open(labels_path, "w", encoding="utf-8") as file:
    for country in label_encoder.classes_:
        file.write(country + "\n")

print("\nTraining complete!")
print("Model saved to:", model_path)
print("Country labels saved to:", labels_path)