from pathlib import Path
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

EXISTING_DATASET = PROJECT_FOLDER / "src" / "filtered_dataset.csv"
NEW_DATASET = PROJECT_FOLDER / "src" / "us_external.csv"
OUTPUT_DATASET = PROJECT_FOLDER / "src" / "combined_dataset.csv"

# Load both datasets
existing_df = pd.read_csv(EXISTING_DATASET)
new_df = pd.read_csv(NEW_DATASET)

# Combine them
combined_df = pd.concat(
    [existing_df, new_df],
    ignore_index=True
)

# Remove accidental duplicate filenames
combined_df = combined_df.drop_duplicates(subset="filename")

# Save the combined dataset
combined_df.to_csv(OUTPUT_DATASET, index=False)

print("Original images:", len(existing_df))
print("New US images:", len(new_df))
print("Combined images:", len(combined_df))
print("\nImages per country:")
print(combined_df["country"].value_counts().to_string())

print(f"\nSaved to: {OUTPUT_DATASET}")