from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn.functional as F


# 1. SETTINGS

PROJECT_FOLDER = Path(__file__).resolve().parents[2]

IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "dataset"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "analysis"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

COUNTS_FILE = OUTPUT_FOLDER / "segmentation_pixel_counts.csv"
SUMMARY_FILE = OUTPUT_FOLDER / "segmentation_category_summary.csv"


# 2. LOAD SEGMENTATION MODEL

print("Loading segmentation model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

# ADE20K category names, indexed by their class ID.
id_to_label = model.config.id2label

print("Number of segmentation categories:", len(id_to_label))


# 3. FIND IMAGES

image_files = [
    path for path in IMAGE_FOLDER.iterdir()
    if path.is_file()
    and path.suffix.lower() in [".png", ".jpg", ".jpeg"]
]


def image_sort_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name.lower())


image_files.sort(key=image_sort_key)

print("Images to analyze:", len(image_files))

if not image_files:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")


# 4. ANALYZE EACH IMAGE

results = []

for index, image_path in enumerate(image_files, start=1):

    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        total_pixels = width * height

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

            logits = F.interpolate(
                outputs.logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False
            )

            predicted_classes = (
                logits.argmax(dim=1)[0].cpu().numpy()
            )

        # Count pixels assigned to each category.
        pixel_counts = np.bincount(
            predicted_classes.flatten(),
            minlength=len(id_to_label)
        )

        row = {
            "filename": image_path.name,
            "width": width,
            "height": height,
            "total_pixels": total_pixels
        }

        for class_id, count in enumerate(pixel_counts):
            category_name = id_to_label.get(
                class_id,
                id_to_label.get(str(class_id), f"class_{class_id}")
            )

            row[f"{class_id}_{category_name}"] = int(count)

        results.append(row)

        # Show the five largest categories for this image.
        largest_classes = pixel_counts.argsort()[-5:][::-1]

        top_categories = ", ".join(
            f"{id_to_label.get(class_id, id_to_label.get(str(class_id), str(class_id)))} "
            f"({pixel_counts[class_id] / total_pixels * 100:.1f}%)"
            for class_id in largest_classes
            if pixel_counts[class_id] > 0
        )

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}: {top_categories}"
        )

        # Save progress every 100 images.
        if index % 100 == 0:
            pd.DataFrame(results).to_csv(COUNTS_FILE, index=False)
            print("Progress saved.")

    except Exception as error:
        print(f"Failed on {image_path.name}: {error}")


# 5. SAVE PIXEL COUNTS

if not results:
    raise ValueError("No images were successfully analyzed.")

counts_df = pd.DataFrame(results)
counts_df.to_csv(COUNTS_FILE, index=False)

print("\nPer-image pixel counts saved to:", COUNTS_FILE)


# 6. SUMMARIZE CATEGORIES ACROSS ALL IMAGES

category_columns = [
    column for column in counts_df.columns
    if column not in ["filename", "width", "height", "total_pixels"]
]

summary = []

all_pixels = counts_df["total_pixels"].sum()

for column in category_columns:
    pixel_count = counts_df[column].sum()
    percentages = counts_df[column] / counts_df["total_pixels"] * 100

    class_id, category_name = column.split("_", 1)

    summary.append({
        "class_id": int(class_id),
        "category": category_name,
        "total_pixels": int(pixel_count),
        "overall_pixel_percentage": pixel_count / all_pixels * 100,
        "average_image_percentage": percentages.mean(),
        "median_image_percentage": percentages.median(),
        "images_with_category": int((counts_df[column] > 0).sum()),
        "percentage_of_images_with_category": (
            (counts_df[column] > 0).mean() * 100
        )
    })

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "overall_pixel_percentage",
    ascending=False
)

summary_df.to_csv(SUMMARY_FILE, index=False)

print("\nTop 20 categories by total pixel coverage:")
print(
    summary_df[
        [
            "class_id",
            "category",
            "overall_pixel_percentage",
            "average_image_percentage",
            "percentage_of_images_with_category"
        ]
    ].head(20).to_string(index=False)
)

print("\nAnalysis finished!")
print("Per-image counts:", COUNTS_FILE)
print("Category summary:", SUMMARY_FILE)