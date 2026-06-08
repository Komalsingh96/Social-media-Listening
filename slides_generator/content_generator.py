import os
import json
from typing import List, Dict, Optional
import openai
from dotenv import load_dotenv

load_dotenv()

class ContentGenerator:
    """Generates slide content using OpenAI's GPT API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        openai.api_key = self.api_key
        self.model = model
    
    def generate_slide_content(self, topic: str, num_slides: int, slide_types: List[str]) -> List[Dict]:
        """Generate content for all slides."""
        prompt = self._create_content_prompt(topic, num_slides)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            
            content = response['choices'][0]['message']['content']
            slides = self._parse_slide_content(content, num_slides)
            return slides
        
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")
    
    def _create_content_prompt(self, topic: str, num_slides: int) -> str:
        """Create prompt for content generation."""
        prompt = f"""Create a professional {num_slides}-slide presentation about '{topic}'.

For each slide provide: slide_number, type, title, and content (bullet points).

Generate exactly {num_slides} slides with engaging content suitable for a professional presentation."""
        return prompt
    
    def _parse_slide_content(self, content: str, num_slides: int) -> List[Dict]:
        """Parse AI-generated content into structured slide data."""
        slides = []
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        for i in range(num_slides):
            slide_type = "title" if i == 0 else ("closing" if i == num_slides - 1 else "content")
            slides.append({
                "slide_number": i + 1,
                "type": slide_type,
                "title": lines[i] if i < len(lines) else f"Slide {i+1}",
                "content": [lines[i+num_slides]] if i+num_slides < len(lines) else []
            })
        
        return slides
