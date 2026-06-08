# Social Media Author Type Classifier

A machine learning-based classifier for tagging social media posts with author types: HCPs (Healthcare Professionals), Patients, Caregivers, and Others.

## Overview

This project provides a comprehensive solution for automatically classifying social media posts based on the author type. It uses natural language processing and machine learning techniques to identify and categorize posts from different user segments in the healthcare and wellness space.

## Features

- **Multi-class Classification**: Classifies posts into 4 categories:
  - HCPs (Healthcare Professionals)
  - Patients
  - Caregivers
  - Others
- **TF-IDF Vectorization**: Converts text to numerical features
- **Multiple ML Models**: Supports Logistic Regression, SVM, and Random Forest
- **Easy Prediction**: Simple API for classifying new posts
- **Training & Evaluation**: Built-in utilities for model training and evaluation
- **Extensible**: Easy to add new features and models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training the Model

```python
from author_classifier import AuthorTypeClassifier

# Initialize classifier
classifier = AuthorTypeClassifier(model_type='logistic_regression')

# Prepare your training data
training_data = [
    {"text": "Just prescribed medication to my patient...", "author_type": "HCP"},
    {"text": "My diagnosis journey started last year...", "author_type": "Patient"},
    {"text": "Taking care of my mom has been challenging...", "author_type": "Caregiver"},
    # ... more examples
]

# Train the model
classifier.train(training_data)

# Save the model
classifier.save_model('author_classifier_model.pkl')
```

### Making Predictions

```python
# Load a trained model
classifier = AuthorTypeClassifier()
classifier.load_model('author_classifier_model.pkl')

# Predict author type for a post
post_text = "I just finished my medical certification exam!"
prediction = classifier.predict(post_text)

print(f"Author Type: {prediction['author_type']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# Batch predictions
posts = [
    "Started my new medication today",
    "As a doctor, I recommend these practices",
    "Helping my father with his recovery"
]

results = classifier.predict_batch(posts)
for i, result in enumerate(results):
    print(f"Post {i+1}: {result['author_type']} (confidence: {result['confidence']:.2%})")
```

## Model Types

### Logistic Regression
- **Pros**: Fast, interpretable, good baseline
- **Cons**: May underperform on complex patterns
- **Best for**: Quick implementations

### Support Vector Machine (SVM)
- **Pros**: Excellent for text classification, handles high-dimensional data
- **Cons**: Slower training on large datasets
- **Best for**: High-accuracy requirements

### Random Forest
- **Pros**: Robust, handles feature interactions well
- **Cons**: Slower prediction time
- **Best for**: Production with balanced speed/accuracy

## Data Format

Training data should be in the following format:

```python
[
    {
        "text": "Post content here",
        "author_type": "HCP"  # One of: HCP, Patient, Caregiver, Others
    },
    # ... more posts
]
```

## Project Structure

```
.
├── author_classifier.py      # Main classifier class
├── feature_extractor.py      # Text feature extraction
├── model_evaluator.py        # Model evaluation utilities
├── sample_data.py            # Sample training data
├── examples.py               # Usage examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Evaluation Metrics

The classifier provides the following evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted positives are actually positive
- **Recall**: How many actual positives are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - feel free to use this project in your applications.

## Support

For questions or issues, please open an issue on GitHub.
