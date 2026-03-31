# Sign Language Alphabet Recognition Project Report

## Title

Sign Language Alphabet Recognition Using MediaPipe Hand Landmarks and Machine Learning

## Abstract

This project presents a real-time sign language alphabet recognition system for static American Sign Language gestures. The system uses a webcam to capture hand images, extracts image-based features from a centered hand region, and classifies the gesture into an alphabet label using a Random Forest model. The project is lightweight, practical for student systems, and suitable for final-year computer vision work because it combines data collection, feature extraction, model training, evaluation, and live deployment.

## Problem Statement

Many people with hearing or speech impairments rely on sign language for communication, but not everyone around them understands sign language. A computer vision system that recognizes sign language letters in real time can help bridge this communication gap and also serve as a building block for larger assistive applications.

## Objectives

- Build a real-time ASL alphabet recognition pipeline.
- Collect or import labeled gesture images.
- Extract meaningful hand features using MediaPipe.
- Train a machine learning classifier on landmark features.
- Evaluate the model using accuracy and confusion matrix analysis.
- Run the system on a webcam feed with live predictions.

## Scope

The current version focuses on static ASL alphabet gestures:

`A B C D E F G H I K L M N O P Q R S T U V W X Y`

The letters `J` and `Z` are excluded because they involve motion and would require a temporal sequence model.

## Methodology

### 1. Data Collection

Images are collected from a webcam or imported from an external dataset. Each class is stored in a separate folder such as `data/raw/A`, `data/raw/B`, and so on.

### 2. Landmark Extraction

The system crops a centered square region from each input image so the user can keep the hand in a predictable position during both training and inference.

### 3. Feature Engineering

The system converts each cropped image into a feature vector by:

- converting the image to grayscale
- resizing to a fixed resolution
- normalizing intensities
- computing gradient-histogram features
- combining those with pixel-intensity features

This produces a compact representation of the hand shape and texture.

### 4. Model Training

A `RandomForestClassifier` is trained on the processed image features. Standard scaling is applied before classification, and the dataset is split into training and testing subsets.

### 5. Evaluation

The system computes:

- classification accuracy
- class-wise precision, recall, and F1-score
- confusion matrix visualization

### 6. Real-Time Prediction

During inference, the webcam stream is processed frame by frame. A centered crop is converted into features, the classifier predicts a class, and temporal smoothing across recent predictions reduces flicker.

## System Architecture

1. Webcam / Image Dataset
2. Center Crop and Image Preprocessing
3. Feature Engineering
4. Random Forest Classification
5. Prediction Display in CLI / GUI

## Tools and Technologies

- Python
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Tkinter for GUI

## Project Structure

- `app.py`: main CLI entrypoint
- `gui.py`: GUI launcher
- `src/asl_recognizer/dataset.py`: sample collection and dataset building
- `src/asl_recognizer/features.py`: feature extraction logic
- `src/asl_recognizer/image_features.py`: image preprocessing and feature extraction
- `src/asl_recognizer/training.py`: model training and evaluation
- `src/asl_recognizer/predictor.py`: live webcam prediction
- `src/asl_recognizer/gui.py`: desktop interface
- `PROJECT_REPORT.md`: report documentation

## Expected Results

With a balanced dataset and consistent hand visibility, the system should correctly recognize most static ASL letters in real time. Performance depends on image quality, lighting, hand orientation, background complexity, and dataset diversity.

## Limitations

- The current system assumes the hand is centered inside the guide box.
- Dynamic gestures like `J` and `Z` are not supported.
- Accuracy depends strongly on the quality of the collected dataset.

## Environment Note

This version removes MediaPipe from the core workflow, so the project is easier to run on newer Python versions. The main requirement is that OpenCV and the machine-learning dependencies install successfully in the chosen environment.

## Future Enhancements

- Add support for dynamic gestures using LSTM or Transformers
- Form full words and sentences from letter sequences
- Save prediction history to text
- Deploy as a web or mobile app
- Support custom sign vocabularies and regional sign systems

## Conclusion

This project demonstrates a complete and practical computer vision application for sign language recognition. It is suitable for academic presentation because it covers the full lifecycle of a machine learning system, from data collection to real-time inference, while remaining lightweight enough for typical student hardware.
