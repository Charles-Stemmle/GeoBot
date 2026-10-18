# Geobot web app

Upload a street-level photo and get country guesses from your three models:

- **Holistic**: the full image
- **Road specialist**: only road, sidewalk, vehicles, poles and signs (ADE20K classes from your road script)
- **Environment specialist**: only vegetation and terrain (your environment script's classes)
- **Consensus**: weighted combination of the three

The page highlights the road and environment regions on the photo (toggle each one, adjust opacity, hover a specialist card to spotlight its region) and shows the top 5 guesses per model.

## Setup

Put this folder inside your project, next to `models/`:

```
your-project/
  models/            <- the .keras files and label .txt files your training scripts saved
  webapp/            <- this folder
```

(If your models live elsewhere, set the environment variable `GEOBOT_MODEL_DIR`.)

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app
```

Open http://127.0.0.1:8000. The first start downloads the SegFormer weights (needs internet once) and loads all models, so give it a minute. Startup prints the names of the segmentation classes it is using for "environment" and "road". Glance at them once to confirm the IDs map to what you intended.

## Tuning the consensus

Edit `weights.json`. It is re-read automatically when it changes, so you can drop in the values from your consensus analysis and just upload another image, with no restart.

```json
{
  "weights": { "holistic": 0.50, "road": 0.25, "environment": 0.25 },
  "temperature": { "holistic": 1.0, "road": 1.0, "environment": 1.0 }
}
```

Weights are normalised to sum to 1. `temperature` is optional per-model calibration (values above 1 soften an overconfident model); leave at 1.0 until you've fitted them on held-out data.

## What happens per upload

1. The image is validated (JPEG/PNG, max 10 MB) and shrunk to at most 640 px, like your training images.
2. SegFormer runs once. Its class map gives both the environment and road masks.
3. Each classifier gets the input it was trained on (full image, road-masked, environment-masked), resized to 224 and scaled with `mobilenet_v2.preprocess_input`.
4. Probabilities are merged **by country name** into the consensus.

## Notes

- **Preprocessing fix:** your `consensus` script divided by 255 (range 0 to 1), but all three models were trained with `mobilenet_v2.preprocess_input` (range -1 to 1). This app uses the training-time preprocessing. Predictions here may differ from your script's, and the weights you tune with the old script may need re-tuning.
- **Countries a model doesn't know** contribute 0 for that country (same behaviour as your script). Training all three on one canonical label list removes this.
- **Very low mask coverage** (under 2% of the image) triggers a warning on the page, since that specialist is then classifying a nearly black image.
- Requests are processed one at a time (a lock guards the models). That is fine for local use. For several simultaneous users, run behind a queue or add workers with enough memory for each.
- Nothing is saved: uploads are processed in memory and discarded.

## Tests

```bash
pip install pytest httpx
python -m pytest test_app.py
```

These cover the consensus math, upload validation and endpoint behaviour with a stubbed pipeline. They do not load TensorFlow or your models.
