from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import json


class ModelEvaluator:
    """
    Evaluates the performance of the author type classifier.
    """
    
    def __init__(self, classifier):
        """
        Initialize the evaluator.
        
        Args:
            classifier (AuthorTypeClassifier): The trained classifier
        """
        self.classifier = classifier
    
    def evaluate(self, test_data):
        """
        Evaluate the classifier on test data.
        
        Args:
            test_data (list): List of dicts with 'text' and 'author_type' keys
        
        Returns:
            dict: Comprehensive evaluation results
        """
        if not test_data:
            raise ValueError("Test data cannot be empty")
        
        # Get predictions
        texts = [item['text'] for item in test_data]
        true_labels = [item['author_type'] for item in test_data]
        
        predictions = self.classifier.predict_batch(texts)
        predicted_labels = [p['author_type'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=self.classifier.AUTHOR_TYPES)
        
        # Per-class metrics
        class_report = classification_report(true_labels, predicted_labels, labels=self.classifier.AUTHOR_TYPES, output_dict=True)
        
        return {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'total_samples': len(test_data)
            },
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_labels': self.classifier.AUTHOR_TYPES,
            'per_class_metrics': class_report,
            'correct_predictions': sum(1 for t, p in zip(true_labels, predicted_labels) if t == p),
            'incorrect_predictions': sum(1 for t, p in zip(true_labels, predicted_labels) if t != p)
        }
    
    def get_misclassified_examples(self, test_data, limit=10):
        """
        Get examples that were misclassified.
        
        Args:
            test_data (list): List of dicts with 'text' and 'author_type' keys
            limit (int): Maximum number of examples to return
        
        Returns:
            list: List of misclassified examples
        """
        texts = [item['text'] for item in test_data]
        true_labels = [item['author_type'] for item in test_data]
        
        predictions = self.classifier.predict_batch(texts)
        predicted_labels = [p['author_type'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        misclassified = []
        for i, (true_label, pred_label, confidence) in enumerate(zip(true_labels, predicted_labels, confidences)):
            if true_label != pred_label:
                misclassified.append({
                    'text': texts[i],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': float(confidence)
                })
        
        return misclassified[:limit]
    
    def print_report(self, test_data):
        """
        Print a formatted evaluation report.
        
        Args:
            test_data (list): List of dicts with 'text' and 'author_type' keys
        """
        results = self.evaluate(test_data)
        
        print("\n" + "="*60)
        print("AUTHOR TYPE CLASSIFIER - EVALUATION REPORT")
        print("="*60)
        
        print("\nOVERALL METRICS:")
        print("-" * 40)
        for metric, value in results['overall_metrics'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("\nCONFUSION MATRIX:")
        print("-" * 40)
        print("Labels:", results['confusion_matrix_labels'])
        for i, row in enumerate(results['confusion_matrix']):
            print(f"{results['confusion_matrix_labels'][i]:10} {row}")
        
        print("\nPER-CLASS METRICS:")
        print("-" * 40)
        for label in self.classifier.AUTHOR_TYPES:
            if label in results['per_class_metrics']:
                metrics = results['per_class_metrics'][label]
                print(f"\n{label}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
                print(f"  Support: {int(metrics['support'])}")
        
        print("\n" + "="*60 + "\n")
