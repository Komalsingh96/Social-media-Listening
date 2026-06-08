from author_classifier import AuthorTypeClassifier
from model_evaluator import ModelEvaluator
from sample_data import get_training_data, get_test_data


def example_basic_training_and_prediction():
    """
    Basic example: Train a model and make predictions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Training and Prediction")
    print("="*60)
    
    # Initialize classifier
    classifier = AuthorTypeClassifier(model_type='logistic_regression')
    
    # Get sample training data
    training_data = get_training_data()
    
    # Train the model
    print("\nTraining the model...")
    train_result = classifier.train(training_data)
    print(f"Training completed!")
    print(f"Model Type: {train_result['model_type']}")
    print(f"Training Accuracy: {train_result['training_accuracy']:.4f}")
    print(f"Samples Trained: {train_result['samples_trained']}")
    
    # Make predictions
    print("\nMaking predictions...")
    test_posts = [
        "I just diagnosed a patient with Type 2 Diabetes and prescribed metformin",
        "My symptoms started three weeks ago. I've been experiencing severe fatigue.",
        "Caring for my elderly mother is rewarding but exhausting",
        "Just finished my morning jog, feeling great!"
    ]
    
    for post in test_posts:
        result = classifier.predict(post)
        print(f"\nPost: {post[:60]}...")
        print(f"Predicted Author Type: {result['author_type']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    return classifier


def example_batch_prediction():
    """
    Example: Batch predictions for multiple posts.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Predictions")
    print("="*60)
    
    # Initialize and train classifier
    classifier = AuthorTypeClassifier(model_type='svm')
    training_data = get_training_data()
    print("\nTraining SVM model...")
    classifier.train(training_data)
    
    # Batch prediction
    posts = [
        "Starting clinical trial participation next week",
        "My mom's treatment is going well, doctors are optimistic",
        "Published new research on disease prevention in healthcare",
        "Feeling much better after the therapy session"
    ]
    
    print("\nMaking batch predictions...")
    results = classifier.predict_batch(posts)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Post: {result['text'][:50]}...")
        print(f"   Author Type: {result['author_type']} (Confidence: {result['confidence']:.2%})")
    
    return classifier


def example_model_evaluation():
    """
    Example: Evaluate model performance on test data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Evaluation")
    print("="*60)
    
    # Initialize and train classifier
    classifier = AuthorTypeClassifier(model_type='random_forest')
    training_data = get_training_data()
    print("\nTraining Random Forest model...")
    classifier.train(training_data)
    
    # Get test data and evaluate
    test_data = get_test_data()
    evaluator = ModelEvaluator(classifier)
    
    print("\nEvaluating model on test data...")
    evaluator.print_report(test_data)
    
    # Get misclassified examples
    print("\nTop Misclassified Examples:")
    print("-" * 60)
    misclassified = evaluator.get_misclassified_examples(test_data, limit=5)
    for i, example in enumerate(misclassified, 1):
        print(f"\n{i}. Text: {example['text'][:60]}...")
        print(f"   True: {example['true_label']}, Predicted: {example['predicted_label']}")
        print(f"   Confidence: {example['confidence']:.2%}")
    
    return classifier


def example_model_persistence():
    """
    Example: Save and load a trained model.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Persistence (Save/Load)")
    print("="*60)
    
    # Train and save model
    classifier1 = AuthorTypeClassifier(model_type='logistic_regression')
    training_data = get_training_data()
    print("\nTraining model...")
    classifier1.train(training_data)
    
    model_path = 'saved_author_classifier.pkl'
    print(f"\nSaving model to {model_path}...")
    save_result = classifier1.save_model(model_path)
    print(f"Save Status: {save_result['status']}")
    
    # Load the model in a new classifier instance
    classifier2 = AuthorTypeClassifier()
    print(f"\nLoading model from {model_path}...")
    load_result = classifier2.load_model(model_path)
    print(f"Load Status: {load_result['status']}")
    
    # Verify loaded model works
    print("\nVerifying loaded model works...")
    test_post = "As a physician, I recommend regular health checkups"
    result = classifier2.predict(test_post)
    print(f"Prediction: {result['author_type']} (Confidence: {result['confidence']:.2%})")
    
    return classifier2


def example_compare_models():
    """
    Example: Compare different model types.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Different Model Types")
    print("="*60)
    
    training_data = get_training_data()
    test_data = get_test_data()
    
    model_types = ['logistic_regression', 'svm', 'random_forest']
    results_summary = []
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        classifier = AuthorTypeClassifier(model_type=model_type)
        classifier.train(training_data)
        
        evaluator = ModelEvaluator(classifier)
        eval_result = evaluator.evaluate(test_data)
        
        results_summary.append({
            'model_type': model_type,
            'accuracy': eval_result['overall_metrics']['accuracy'],
            'precision': eval_result['overall_metrics']['precision'],
            'recall': eval_result['overall_metrics']['recall'],
            'f1_score': eval_result['overall_metrics']['f1_score']
        })
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model Type':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    
    for result in results_summary:
        print(f"{result['model_type']:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1_score']:<12.4f}")


if __name__ == "__main__":
    # Run all examples
    example_basic_training_and_prediction()
    example_batch_prediction()
    example_model_evaluation()
    example_model_persistence()
    example_compare_models()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
