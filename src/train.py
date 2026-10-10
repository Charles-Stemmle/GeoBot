
from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parent.parent


DATASET_FILE = PROJECT_FOLDER / "src" / "combined_dataset.csv"
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "dataset"
MODEL_FOLDER = PROJECT_FOLDER / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
RANDOM_SEED = 42


# 2. LOAD THE DATASET

print("Loading dataset...")

data = pd.read_csv(DATASET_FILE)

print("Total images:", len(data))
print("Number of countries:", data["country"].nunique())


# 3. SPLIT TRAINING AND VALIDATION DATA


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



# 5. LOAD AND PROCESS IMAGES



def load_image(filename, label):
    # Convert the TensorFlow filename tensor into a full image path.
    image_path = tf.strings.join([
        str(IMAGE_FOLDER) + "\\",
        filename
    ])

    # Read the image file.
    image = tf.io.read_file(image_path)

    # Decode the PNG into RGB pixel data.
    image = tf.io.decode_png(image, channels=3)

    # Resize to 224 x 224 pixels.
    image = tf.image.resize(image, IMAGE_SIZE)

    # Prepare the pixel values for MobileNetV2.
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# -----------------------------
# 6. CREATE THE INPUT PIPELINE
# -----------------------------

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

train_dataset = create_dataset(
    train,
    y_train,
    shuffle=True
)

validation_dataset = create_dataset(
    validation,
    y_validation
)


# -----------------------------
# 7. CHECK THE TENSOR SHAPES
# -----------------------------

print("\nChecking image tensors...")

for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First image shape:", images[0].shape)
    print("First pixel:", images[0, 0, 0].numpy())


# -----------------------------
# 8. BUILD THE MODEL
# -----------------------------

print("\nBuilding MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers initially.
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


# -----------------------------
# 9. CONFIGURE TRAINING
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# 10. TRAIN THE MODEL
# -----------------------------

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# -----------------------------
# 11. EVALUATE THE MODEL
# -----------------------------

print("\nEvaluating model...")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
)

print(f"Validation loss: {validation_loss:.4f}")
print(f"Validation accuracy: {validation_accuracy:.2%}")


# -----------------------------
# 12. SAVE THE MODEL
# -----------------------------

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

model_path = MODEL_FOLDER / "geobot_country_classifier.keras"

model.save(model_path)

# Save the country names in the same order as the model's outputs.
labels_path = MODEL_FOLDER / "country_labels.txt"

with open(labels_path, "w", encoding="utf-8") as file:
    for country in label_encoder.classes_:
        file.write(country + "\n")

print("\nTraining complete!")
print("Model saved to:", model_path)
print("Country labels saved to:", labels_path)