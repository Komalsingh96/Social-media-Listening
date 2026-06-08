import re
from collections import Counter


class FeatureExtractor:
    """
    Extracts linguistic and domain-specific features from social media posts.
    """
    
    # Keywords associated with each author type
    HCP_KEYWORDS = [
        'doctor', 'physician', 'nurse', 'patient', 'medication', 'diagnosis',
        'treatment', 'clinical', 'medical', 'health', 'therapy', 'prescription',
        'hospital', 'clinic', 'surgery', 'examination', 'disease', 'condition',
        'symptoms', 'practice', 'healthcare', 'professional', 'certificate',
        'degree', 'license', 'certified', 'residency', 'fellowship'
    ]
    
    PATIENT_KEYWORDS = [
        'diagnosed', 'symptoms', 'pain', 'suffering', 'treatment', 'medicine',
        'doctor', 'hospital', 'appointment', 'condition', 'disease', 'illness',
        'recovery', 'healing', 'experience', 'journey', 'struggle', 'battle',
        'my condition', 'been diagnosed', 'my diagnosis', 'my health'
    ]
    
    CAREGIVER_KEYWORDS = [
        'care', 'caregiver', 'family', 'loved one', 'parent', 'mother', 'father',
        'child', 'children', 'spouse', 'partner', 'support', 'helping', 'helping',
        'assisted', 'taking care', 'look after', 'responsible', 'challenging',
        'difficult', 'exhausting', 'patience', 'compassion', 'dedication'
    ]
    
    def __init__(self):
        pass
    
    def extract_features(self, text):
        """
        Extract various features from text.
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Basic text features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        features['sentence_count'] = len(re.split(r'[.!?]', text)) - 1
        
        # Keyword features
        text_lower = text.lower()
        features['hcp_keyword_count'] = self._count_keywords(text_lower, self.HCP_KEYWORDS)
        features['patient_keyword_count'] = self._count_keywords(text_lower, self.PATIENT_KEYWORDS)
        features['caregiver_keyword_count'] = self._count_keywords(text_lower, self.CAREGIVER_KEYWORDS)
        
        # Linguistic features
        features['has_first_person'] = self._has_first_person_singular(text)
        features['has_plural_first_person'] = self._has_first_person_plural(text)
        features['has_family_reference'] = self._has_family_reference(text)
        features['has_professional_tone'] = self._has_professional_tone(text)
        features['has_emotion_words'] = self._count_emotion_words(text)
        
        return features
    
    def _count_keywords(self, text, keywords):
        """
        Count occurrences of keywords in text.
        
        Args:
            text (str): Lowercase text
            keywords (list): List of keywords
        
        Returns:
            int: Count of keyword occurrences
        """
        count = 0
        for keyword in keywords:
            count += len(re.findall(r'\b' + keyword + r'\b', text))
        return count
    
    def _has_first_person_singular(self, text):
        """
        Check if text contains first person singular pronouns.
        """
        return bool(re.search(r'\b(i|me|my|mine)\b', text.lower()))
    
    def _has_first_person_plural(self, text):
        """
        Check if text contains first person plural pronouns.
        """
        return bool(re.search(r'\b(we|us|our|ours)\b', text.lower()))
    
    def _has_family_reference(self, text):
        """
        Check if text mentions family relationships.
        """
        family_terms = ['mother', 'father', 'parent', 'son', 'daughter', 'child', 'wife', 'husband', 'sibling', 'brother', 'sister', 'family']
        text_lower = text.lower()
        return any(term in text_lower for term in family_terms)
    
    def _has_professional_tone(self, text):
        """
        Check if text has a professional tone.
        """
        professional_words = ['recommend', 'suggest', 'clinical', 'evidence', 'research', 'study', 'practice', 'protocol', 'guidelines']
        text_lower = text.lower()
        return sum(1 for word in professional_words if word in text_lower)
    
    def _count_emotion_words(self, text):
        """
        Count emotion-related words in text.
        """
        emotion_words = ['love', 'hate', 'amazing', 'terrible', 'horrible', 'wonderful', 'struggling', 'fighting', 'suffering', 'grateful', 'blessed', 'sad', 'happy', 'angry', 'frustrated']
        text_lower = text.lower()
        return sum(1 for word in emotion_words if word in text_lower)
