def get_training_data():
    """
    Returns sample training data for the author type classifier.
    """
    return [
        # HCP (Healthcare Professional) examples
        {"text": "Just completed a successful surgery this morning. Great teamwork with the surgical team!", "author_type": "HCP"},
        {"text": "Presented new clinical findings at the medical conference today. Really excited about the research outcomes.", "author_type": "HCP"},
        {"text": "As a cardiologist, I recommend regular checkups for patients with family history of heart disease.", "author_type": "HCP"},
        {"text": "Started implementing new evidence-based treatment protocols at our clinic this week.", "author_type": "HCP"},
        {"text": "My nursing degree taught me the importance of patient care and compassion in healthcare.", "author_type": "HCP"},
        {"text": "Attended a professional development conference on the latest treatment methodologies.", "author_type": "HCP"},
        {"text": "Diagnostic imaging revealed interesting findings in today's patient assessment.", "author_type": "HCP"},
        {"text": "Published my research on disease prevention in a peer-reviewed medical journal.", "author_type": "HCP"},
        
        # Patient examples
        {"text": "Been struggling with my diagnosis for the past year. Finally starting to feel better after treatment.", "author_type": "Patient"},
        {"text": "My symptoms started three weeks ago with severe fatigue and joint pain.", "author_type": "Patient"},
        {"text": "Just got the results back and the doctor said I'm responding well to the medication.", "author_type": "Patient"},
        {"text": "The condition has really affected my daily life, but I'm learning to manage it better.", "author_type": "Patient"},
        {"text": "Started my recovery journey today. Hoping for the best outcome from the treatment plan.", "author_type": "Patient"},
        {"text": "Living with this chronic illness has been challenging, but support groups help a lot.", "author_type": "Patient"},
        {"text": "My health journey taught me the importance of taking care of myself.", "author_type": "Patient"},
        {"text": "The side effects from the new medication are manageable compared to the previous one.", "author_type": "Patient"},
        
        # Caregiver examples
        {"text": "Taking care of my elderly mother is rewarding but emotionally exhausting.", "author_type": "Caregiver"},
        {"text": "My dad's health condition requires 24/7 care and attention from our family.", "author_type": "Caregiver"},
        {"text": "Being a caregiver has taught me patience and the true meaning of love.", "author_type": "Caregiver"},
        {"text": "Helping my spouse through recovery has been one of the most challenging periods of my life.", "author_type": "Caregiver"},
        {"text": "My child's illness brought our family closer together despite the difficulties.", "author_type": "Caregiver"},
        {"text": "Looking after a loved one with a chronic condition requires dedication and support.", "author_type": "Caregiver"},
        {"text": "As a family caregiver, I've learned to balance work and caring responsibilities.", "author_type": "Caregiver"},
        {"text": "My mother's treatment journey has required me to become an advocate for her health needs.", "author_type": "Caregiver"},
        
        # Others examples
        {"text": "Just finished my morning jog and had a great workout at the gym!", "author_type": "Others"},
        {"text": "Excited to try this new healthy recipe I found online.", "author_type": "Others"},
        {"text": "Love spending time outdoors, it's great for overall wellness.", "author_type": "Others"},
        {"text": "Recently started meditating to manage stress in my daily life.", "author_type": "Others"},
        {"text": "Can't wait for the weekend to relax and spend time with friends.", "author_type": "Others"},
        {"text": "Following a new fitness routine and feeling great!", "author_type": "Others"},
        {"text": "Just finished reading an interesting book about wellness and lifestyle.", "author_type": "Others"},
        {"text": "Enjoying a peaceful day at the beach with family.", "author_type": "Others"},
    ]


def get_test_data():
    """
    Returns sample test data for evaluating the classifier.
    """
    return [
        # HCP test examples
        {"text": "Reviewing patient charts and preparing for tomorrow's rounds.", "author_type": "HCP"},
        {"text": "My medical practice focuses on preventive healthcare and patient education.", "author_type": "HCP"},
        
        # Patient test examples
        {"text": "Feeling anxious about the upcoming medical procedure.", "author_type": "Patient"},
        {"text": "Recovery has been slower than expected but I'm staying positive.", "author_type": "Patient"},
        
        # Caregiver test examples
        {"text": "Juggling work and caring for my parents is quite demanding.", "author_type": "Caregiver"},
        {"text": "I've become more compassionate through my caregiving experience.", "author_type": "Caregiver"},
        
        # Others test examples
        {"text": "Just started a new fitness class and loving it!", "author_type": "Others"},
        {"text": "Weekend hiking adventure was absolutely amazing!", "author_type": "Others"},
    ]
