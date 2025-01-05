# SCT_ML_3
This project implements a Support Vector Machine (SVM) to classify images of cats and dogs from the popular Kaggle dataset.
# Task 03: Cat vs. Dog Classification Using SVM

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs from the popular Kaggle dataset.

## Overview

The goal of this project is to use machine learning techniques, specifically SVM, to accurately distinguish between images of cats and dogs. This involves:

1. Data preprocessing and preparation.
2. Feature extraction.
3. Training an SVM model.
4. Evaluating the model's performance.

## Dataset

We use the **Dog/Cat Dataset** from Kaggle, which contains thousands of labeled images of cats and dogs.  
[Link to the dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

## Steps Involved

1. **Dataset Preparation**:
   - Download and extract the dataset.
   - Split into training and test sets.

2. **Data Preprocessing**:
   - Resize images to a uniform size.
   - Convert images to grayscale or extract color features.
   - Normalize pixel values for better model performance.

3. **Feature Extraction**:
   - Use techniques like HOG (Histogram of Oriented Gradients) or flatten pixel values to extract meaningful features.

4. **Model Training**:
   - Train an SVM classifier using the extracted features.
   - Tune hyperparameters like the kernel type (e.g., linear, RBF) and regularization parameter.

5. **Evaluation**:
   - Evaluate the model using accuracy, precision, recall, and F1 score.
   - Visualize the confusion matrix.

## Requirements

- Python 3.7+
- Required Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `opencv-python` (for image processing)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-vs-dog-svm.git
   cd cat-vs-dog-svm
   ```

2. Run the notebook:
   ```bash
   jupyter notebook SCT_ML_3.ipynb
   ```

3. Follow the instructions in the notebook to preprocess data, train the model, and evaluate results.

## Results

- The SVM classifier achieved an accuracy of **XX%** on the test set.
- Example predictions:
  - 🐶 Correctly classified dog image.
  - 🐱 Correctly classified cat image.

## Future Improvements

- Use more advanced feature extraction methods like CNN-based embeddings.
- Experiment with other ML algorithms.
- Deploy the model using Flask or FastAPI.

## Acknowledgments

- Thanks to Kaggle for providing the **Dogs vs. Cats** dataset.
- This task is part of my **Machine Learning Internship at SkillCraft Technology**.
