# Sign Language Alphabet Recognition

This is a complete computer vision project for recognizing static American Sign Language alphabet gestures from a webcam. It uses a direct image-classification pipeline based on cropped hand images and machine learning, which makes it easier to run on newer Python versions.

## What this project includes

- Webcam sample collection for any ASL letter
- Dataset builder that converts images into image-based features
- Model training with evaluation metrics and confusion matrix output
- Live webcam prediction with on-screen confidence display
- Desktop GUI for running the workflow
- Environment checker for dependency readiness
- A clean structure that is easy to extend into a final year project

## Python compatibility

The main project flow no longer depends on MediaPipe. It is designed to run on newer Python versions as long as `opencv-python`, `numpy`, `scikit-learn`, `joblib`, and `matplotlib` install successfully.

## Important note

This project is designed for **static ASL letters**. It works best for:

`A B C D E F G H I K L M N O P Q R S T U V W X Y`

The letters `J` and `Z` are usually dynamic gestures that depend on motion over time, so they are not included in the default label set.

## Project structure

```text
project/
+-- app.py
+-- gui.py
+-- requirements.txt
+-- README.md
+-- PROJECT_REPORT.md
+-- artifacts/
|   +-- asl_classifier.joblib
|   +-- confusion_matrix.png
|   `-- training_metrics.json
+-- data/
|   +-- raw/
|   |   +-- A/
|   |   +-- B/
|   |   `-- ...
|   `-- processed/
|       +-- asl_image_dataset.npz
|       `-- dataset_summary.json
`-- src/
    `-- asl_recognizer/
        +-- __init__.py
        +-- config.py
        +-- dataset.py
        +-- image_features.py
        +-- features.py
        +-- hand_tracking.py
        +-- predictor.py
        `-- training.py
```

## Setup

1. Create a virtual environment.
2. Install the dependencies.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check the environment anytime with:

```powershell
python app.py doctor
```

## How to run

### 1. Collect your own training images

Example for collecting 200 images of the sign `A`:

```powershell
python app.py collect --label A --samples 200
```

Repeat this for each letter you want to train.

Recommended minimum:

- 150 to 300 images per letter
- Different lighting conditions
- Slightly different hand positions
- Plain background first, then more realistic backgrounds

### Quick demo mode

If you want to verify the full pipeline immediately without downloading a real dataset, generate a synthetic demo dataset:

```powershell
python app.py generate-demo-data --samples-per-class 20
python app.py build-dataset
python app.py train
```

This is only for testing that the project runs end to end. It is **not** suitable for real sign-language recognition from a webcam.

### 2. Or use an image dataset

If you already have an ASL alphabet image dataset, organize it like this:

```text
data/raw/
+-- A/
+-- B/
+-- C/
`-- ...
```

Each folder should contain images for that letter.

The dataset builder also supports a split layout like:

```text
downloaded_dataset/
+-- Train/
|   +-- A/
|   +-- B/
|   `-- ...
`-- Test/
    +-- A/
    +-- B/
    `-- ...
```

Then run:

```powershell
python app.py build-dataset --data-dir downloaded_dataset
```

### Recommended datasets

Best fit for this project:

- Mendeley `American Sign Language Alphabet Dataset` (published December 4, 2025). It already provides `Train` and `Test` folders with per-letter subfolders, which matches this project's builder well.
  Source: [Mendeley dataset](https://data.mendeley.com/datasets/jdyksv2jhh/1)

Good large alternatives:

- `SignAlphaSet` on Mendeley, published March 10, 2025, with 26,000 ASL alphabet images in 26 folders.
  Source: [Mendeley SignAlphaSet](https://data.mendeley.com/datasets/8fmvr9m98w/1)
- `Images of American Sign Language (ASL) Alphabet Gestures` on Mendeley, updated December 16, 2024, with 210,000 images across 28 classes.
  Source: [Mendeley 210k dataset](https://data.mendeley.com/datasets/48dg9vhmyk/2)

Notes:

- This project targets static letters, so `J` and `Z` are skipped by default.
- Some public datasets also include classes like `DEL`, `SPACE`, or `NOTHING`; those are ignored by the current trainer unless you extend the label set.

### 3. Build the image dataset

```powershell
python app.py build-dataset
```

This creates:

- `data/processed/asl_image_dataset.npz`
- `data/processed/dataset_summary.json`

### 4. Train the model

```powershell
python app.py train
```

This creates:

- `artifacts/asl_classifier.joblib`
- `artifacts/training_metrics.json`
- `artifacts/confusion_matrix.png`

### 5. Run live prediction

```powershell
python app.py predict
```

Press `Q` to close the webcam window.

### 6. Launch the GUI

```powershell
python gui.py
```

Or:

```powershell
python app.py gui
```

The GUI lets you:

- collect samples
- build the dataset
- train the model
- run live prediction
- inspect environment readiness

## Suggested full workflow for a final year project

1. Collect or import a dataset for 24 static ASL letters.
2. Build image features from cropped hand images.
3. Train the Random Forest classifier.
4. Evaluate accuracy and confusion matrix.
5. Run real-time webcam inference.
6. Add report sections for problem statement, methodology, results, and future improvements.

## How the model works

1. The webcam or dataset image is cropped around a centered square guide box.
2. The cropped image is converted to grayscale, normalized, and resized.
3. Pixel and gradient-histogram features are extracted from the image.
4. A `RandomForestClassifier` learns to map those features to ASL letters.
5. During webcam inference, predictions are smoothed across recent frames to reduce flicker.

## Ideas for improving this project later

- Add sentence formation by collecting predicted letters over time
- Add a GUI with Tkinter or Streamlit
- Train a deep learning model on cropped hand images
- Support dynamic letters `J` and `Z` using video sequence models
- Add Hindi or custom sign vocabulary
- Export the model into a mobile-friendly app

## Reference structure used for this implementation

This implementation follows a practical computer vision pipeline:

- image folders per class
- image preprocessing
- feature engineering
- supervised classifier training
- real-time inference from webcam

This approach is intentionally chosen because it is lighter and easier to run on normal college laptops than training a full CNN from scratch.
