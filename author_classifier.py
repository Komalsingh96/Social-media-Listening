import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from feature_extractor import FeatureExtractor


class AuthorTypeClassifier:
    """
    A classifier for tagging social media posts with author types:
    HCPs (Healthcare Professionals), Patients, Caregivers, and Others.
    """
    
    AUTHOR_TYPES = ['HCP', 'Patient', 'Caregiver', 'Others']
    
    def __init__(self, model_type='logistic_regression', max_features=5000):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model - 'logistic_regression', 'svm', or 'random_forest'
            max_features (int): Maximum number of features for TF-IDF vectorizer
        """
        self.model_type = model_type
        self.max_features = max_features
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.is_trained = False
        
    def _get_model(self):
        """
        Get the appropriate ML model based on model_type.
        
        Returns:
            Scikit-learn model instance
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        elif self.model_type == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, training_data):
        """
        Train the classifier on the provided data.
        
        Args:
            training_data (list): List of dicts with 'text' and 'author_type' keys
        
        Returns:
            dict: Training results including accuracy and model info
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Extract texts and labels
        texts = [item['text'] for item in training_data]
        labels = [item['author_type'] for item in training_data]
        
        # Validate author types
        for label in labels:
            if label not in self.AUTHOR_TYPES:
                raise ValueError(f"Invalid author type: {label}. Must be one of {self.AUTHOR_TYPES}")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Create pipeline with TF-IDF vectorizer and model
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ('model', self._get_model())
        ])
        
        # Train the model
        self.pipeline.fit(texts, encoded_labels)
        self.is_trained = True
        
        # Calculate training accuracy
        train_accuracy = self.pipeline.score(texts, encoded_labels)
        
        return {
            'status': 'success',
            'model_type': self.model_type,
            'training_accuracy': train_accuracy,
            'samples_trained': len(training_data),
            'classes': self.AUTHOR_TYPES
        }
    
    def predict(self, text):
        """
        Predict the author type for a single post.
        
        Args:
            text (str): The social media post text
        
        Returns:
            dict: Prediction result with author_type and confidence
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Get prediction and probabilities
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        # Convert encoded label back to author type
        author_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        return {
            'author_type': author_type,
            'confidence': confidence,
            'text': text
        }
    
    def predict_batch(self, texts):
        """
        Predict author types for multiple posts.
        
        Args:
            texts (list): List of post texts
        
        Returns:
            list: List of prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        results = []
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        for i, text in enumerate(texts):
            author_type = self.label_encoder.inverse_transform([predictions[i]])[0]
            confidence = float(np.max(probabilities[i]))
            
            results.append({
                'author_type': author_type,
                'confidence': confidence,
                'text': text
            })
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path where to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model")
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'author_types': self.AUTHOR_TYPES
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return {'status': 'success', 'message': f'Model saved to {filepath}'}
    
    def load_model(self, filepath):
        """
        Load a previously trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        return {'status': 'success', 'message': f'Model loaded from {filepath}'}
    
    def get_feature_importance(self):
        """
        Get feature importance if using a model that supports it (Random Forest).
        
        Returns:
            dict: Feature importance information or None if not available
        """
        if self.model_type != 'random_forest':
            return {'status': 'not_available', 'reason': f'{self.model_type} does not support feature importance'}
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        model = self.pipeline.named_steps['model']
        vectorizer = self.pipeline.named_steps['tfidf']
        
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        
        # Get top 20 features
        top_indices = np.argsort(importances)[-20:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
        
        return {
            'status': 'success',
            'top_features': top_features,
            'model_type': self.model_type
        }
