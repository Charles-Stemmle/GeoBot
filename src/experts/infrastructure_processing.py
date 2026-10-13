
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn.functional as F


# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "dataset"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "photos" / "infrastructure"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
INFRASTRUCTURE_IDS = [
    0,   # wall
    1,   # building
    14,  # door
    25,  # house
    48,  # skyscraper
]


print("Loading segmentation model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

# Find image files and sort them by numeric filename when possible.
image_files = [
    path for path in IMAGE_FOLDER.iterdir()
    if path.is_file() and path.suffix.lower() in [".png", ".jpg", ".jpeg"]
]

def image_sort_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name.lower())

image_files.sort(key=image_sort_key)


print(f"Images to process: {len(image_files)}")

for index, image_path in enumerate(image_files, start=1):
    output_path = OUTPUT_FOLDER / f"{image_path.stem}.png"

    if output_path.exists():
        print(f"[{index}/{len(image_files)}] Already processed: {image_path.name}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        original = np.array(image)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

            logits = F.interpolate(
                outputs.logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False
            )

            predicted_classes = logits.argmax(dim=1)[0].numpy()

        infrastructure_mask = np.isin(predicted_classes, INFRASTRUCTURE_IDS)

        result = original.copy()
        result[~infrastructure_mask] = 0

        Image.fromarray(result).save(output_path)

        percentage = infrastructure_mask.mean() * 100

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}: {percentage:.1f}% infrastructure"
        )

    except Exception as error:
        print(f"Failed on {image_path.name}: {error}")

print("\nBatch segmentation finished.")
print("Results saved to:", OUTPUT_FOLDER)
