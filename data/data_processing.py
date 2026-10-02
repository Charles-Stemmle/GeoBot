import pandas as pd
from pathlib import Path
from geopy.geocoders import Nominatim

# Folder containing the 10,000 images
IMAGE_FOLDER = Path("data/photos/dataset")

# Coordinate CSV
CSV_FILE = Path("coords.csv")

# Read coordinates
coords = pd.read_csv(
    CSV_FILE,
    header=None,
    names=["latitude", "longitude"]
)

# Find all PNG images
image_files = [
    p for p in IMAGE_FOLDER.iterdir()
    if p.suffix.lower() == ".png"
]

# Sort images by the number in their filename
image_files = sorted(
    image_files,
    key=lambda p: int(p.stem)
)

print(f"Images found: {len(image_files)}")
print(f"Coordinates found: {len(coords)}")

# Make sure the numbers match
if len(image_files) != len(coords):
    raise ValueError(
        f"Number of images ({len(image_files)}) "
        f"does not match coordinates ({len(coords)})"
    )

# Add filenames
coords["filename"] = [p.name for p in image_files]

# Check if dataset already exists
if Path("dataset.csv").exists():
    dataset = pd.read_csv("dataset.csv")
else:
    dataset = coords[["filename", "latitude", "longitude"]].copy()
    dataset["country"] = ""

geolocator = Nominatim(user_agent="geoguesser_ml")

n = 0

# Find the first row without a country
for index, row in dataset.iterrows():
    if pd.isna(row["country"]) or row["country"] == "":
        n = index
        break

print(f"Starting at row {n}")

# Process the next 200 rows
batch_amount = 200 

for i in range(n, n + batch_amount):

    location = geolocator.reverse(
        (coords.loc[i, "latitude"], coords.loc[i, "longitude"])
    )

    geo_dict = location.raw
    dataset.loc[i, "country"] = geo_dict["address"].get("country", "")
    print(f"Processed row {i}: {dataset.loc[i, 'country']}")

# Save dataset
dataset.to_csv("dataset.csv", index=False)

print(dataset.head())