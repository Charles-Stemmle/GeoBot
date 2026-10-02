import pandas as pd
from pathlib import Path
import geopandas as gpd


# File locations
IMAGE_FOLDER = Path("data/photos/dataset")
CSV_FILE = Path("coords.csv")
COUNTRY_FILE = Path(
    "data/raw/ne_110m_admin_0_countries.shp"
)


# Load coordinates
coords = pd.read_csv(
    CSV_FILE,
    header=None,
    names=["latitude", "longitude"]
)


# Find images
image_files = [
    p
    for p in IMAGE_FOLDER.iterdir()
    if p.suffix.lower() == ".png"
]


# Sort images numerically
image_files = sorted(
    image_files,
    key=lambda p: int(p.stem)
)


print(f"Images found: {len(image_files)}")
print(f"Coordinates found: {len(coords)}")


# Make sure the number of images and coordinates match
if len(image_files) != len(coords):
    raise ValueError(
        f"Number of images ({len(image_files)}) "
        f"does not match coordinates ({len(coords)})"
    )


# Add filenames to the coordinate data
coords["filename"] = [
    p.name
    for p in image_files
]


# Load the country boundaries
print("\nLoading country boundaries...")

world = gpd.read_file(COUNTRY_FILE)

print("Country dataset loaded.")


# Create geographic points from all coordinates
print("Creating coordinate points...")

points = gpd.GeoDataFrame(
    coords,
    geometry=gpd.points_from_xy(
        coords["longitude"],
        coords["latitude"]
    ),
    crs="EPSG:4326"
)


# Match each coordinate to a country
print("Finding countries...")

result = gpd.sjoin(
    points,
    world[["ADMIN", "geometry"]],
    how="left",
    predicate="within"
)


# Add country names to our dataset
coords["country"] = result["ADMIN"].values


# Show some results
print("\nFirst 10 results:")

print(
    coords[
        [
            "filename",
            "latitude",
            "longitude",
            "country"
        ]
    ].head(10)
)


# Count successful matches
matched = coords["country"].notna().sum()
unmatched = coords["country"].isna().sum()

print("\nResults:")
print(f"Matched:   {matched}")
print(f"Unmatched: {unmatched}")


# Save the completed dataset
coords.to_csv(
    "dataset.csv",
    index=False
)

print("\nSaved dataset.csv")

