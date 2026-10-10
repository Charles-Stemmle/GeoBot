from pathlib import Path
import csv

from datasets import load_dataset

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
IMAGE_FOLDER = PROJECT_FOLDER / "data" / "photos" / "dataset"
OUTPUT_CSV = PROJECT_FOLDER / "src" / "us_external.csv"

MAX_IMAGES = 200

IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

print("Connecting to StreetView Global...")

dataset = load_dataset(
    "Reubencf/streetview-global",
    split="train",
    streaming=True
)

downloaded = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "country"]
    )
    writer.writeheader()

    for row in dataset:
        region = str(row.get("region", "")).strip()

        # Only accept specific USA city labels, not all of North America.
        if not region.startswith("USA - "):
            continue

        image = row.get("image")

        if image is None:
            continue

        filename = f"mapillary_us_{downloaded + 1:04d}.png"
        image_path = IMAGE_FOLDER / filename

        # Resize to reduce storage requirements.
        image = image.convert("RGB")
        image.thumbnail((640, 640))
        image.save(image_path, format="PNG")

        writer.writerow({
            "filename": filename,
            "country": "United States"
        })

        downloaded += 1
        print(f"Saved {downloaded}/{MAX_IMAGES}: {region}")

        if downloaded >= MAX_IMAGES:
            break

print(f"\nFinished. Downloaded {downloaded} US images.")
print(f"Image folder: {IMAGE_FOLDER}")
print(f"Labels saved to: {OUTPUT_CSV}")