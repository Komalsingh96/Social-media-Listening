from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from typing import List, Dict

class SlideDesigner:
    """Handles slide design, formatting, and layout."""
    
    def __init__(self, presentation: Presentation, theme_colors: Dict):
        self.presentation = presentation
        self.theme_colors = theme_colors
    
    def create_title_slide(self, title: str, subtitle: str = "") -> None:
        """Create a title slide."""
        slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*self.theme_colors['primary'])
        
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(1.5))
        title_frame = title_box.text_frame
        title_frame.text = title
        title_frame.word_wrap = True
        
        for paragraph in title_frame.paragraphs:
            paragraph.font.size = Pt(54)
            paragraph.font.bold = True
            paragraph.font.color.rgb = RGBColor(255, 255, 255)
            paragraph.alignment = PP_ALIGN.CENTER
        
        if subtitle:
            subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.7), Inches(9), Inches(1))
            subtitle_frame = subtitle_box.text_frame
            subtitle_frame.text = subtitle
            
            for paragraph in subtitle_frame.paragraphs:
                paragraph.font.size = Pt(32)
                paragraph.font.color.rgb = RGBColor(200, 200, 200)
                paragraph.alignment = PP_ALIGN.CENTER
    
    def create_content_slide(self, title: str, content_points: List[str]) -> None:
        """Create a content slide with bullet points."""
        slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*self.theme_colors['background'])
        
        self._add_slide_title(slide, title)
        
        content_box = slide.shapes.add_textbox(Inches(0.75), Inches(1.5), Inches(8.5), Inches(4.5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True
        
        for i, point in enumerate(content_points):
            if i == 0:
                p = text_frame.paragraphs[0]
            else:
                p = text_frame.add_paragraph()
            
            p.text = point
            p.font.size = Pt(24)
            p.font.color.rgb = RGBColor(*self.theme_colors['text'])
            p.space_before = Pt(6)
            p.space_after = Pt(6)
    
    def create_closing_slide(self, title: str = "Thank You", contact: str = "") -> None:
        """Create a closing/thank you slide."""
        slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*self.theme_colors['primary'])
        
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
        title_frame = title_box.text_frame
        title_frame.text = title
        
        for paragraph in title_frame.paragraphs:
            paragraph.font.size = Pt(54)
            paragraph.font.bold = True
            paragraph.font.color.rgb = RGBColor(255, 255, 255)
            paragraph.alignment = PP_ALIGN.CENTER
    
    def _add_slide_title(self, slide, title: str) -> None:
        """Add a title to a slide with theme styling."""
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
        title_frame = title_box.text_frame
        title_frame.text = title
        
        for paragraph in title_frame.paragraphs:
            paragraph.font.size = Pt(40)
            paragraph.font.bold = True
            paragraph.font.color.rgb = RGBColor(*self.theme_colors['accent'])
            paragraph.alignment = PP_ALIGN.LEFT
