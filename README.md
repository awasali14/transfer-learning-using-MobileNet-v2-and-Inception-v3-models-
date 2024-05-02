# Transfer-learning-using-MobileNet-v2-and-Inception-v3-models



## Project Overview

This project utilizes TensorFlow and TensorFlow Hub to create and train image classification models leveraging transfer learning techniques. The core focus is on classifying images from the `tf_flowers` dataset, which provides a diverse set of flower photographs. The project showcases how to apply transfer learning from pre-trained models like MobileNet V2 and Inception V3 to significantly reduce the computational burden that comes with training a deep learning model from scratch.

## Motivation

The motivation behind this project was to explore the capabilities of transfer learning in efficiently solving image classification tasks with high accuracy, even with limited computational resources. By leveraging pre-trained models, the project addresses the challenge of requiring extensive computational power and training data to develop effective image classifiers.

## Problem Solved

This project tackles the problem of efficiently classifying images into predefined categories. It demonstrates the effectiveness of using feature vectors extracted from models trained on large datasets to enhance learning outcomes on a much smaller dataset. This approach significantly reduces the time and resources required for training while maintaining high accuracy levels.

## What Was Learned

Throughout the project, key learnings include:

- How to implement TensorFlow and TensorFlow Hub for transfer learning.
- The process of data preprocessing in TensorFlow to suit model requirements.
- The dynamics of training neural networks with pre-trained models as a starting point.
- Techniques for evaluating model performance using accuracy metrics and visualizing training/validation results to diagnose model behavior.

## Technical Details

### Requirements

- Python 3.8+
- TensorFlow 2.0
- TensorFlow Hub
- TensorFlow Datasets
- Matplotlib
- Numpy


### Data

The project uses the `tf_flowers` dataset available through TensorFlow Datasets, automatically split into training and validation sets.

### Models

- **MobileNet V2 Feature Extractor**: A lightweight model ideal for mobile and edge devices, used here for extracting robust features.

- **Inception V3 Feature Extractor**: A more complex model that provides higher accuracy at the cost of increased computational complexity.

## Results

The project outputs the training and validation accuracy and loss, which are visualized using Matplotlib. This visualization helps in understanding the model's training progress and its performance on unseen data.

### Predictions

The script outputs images with predictions where the predicted and true labels are displayed. Correct predictions are highlighted in blue, and incorrect ones in red, providing a quick visual assessment of the model's performance.




