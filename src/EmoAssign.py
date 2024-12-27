from transformers import pipeline
import torch



class EmoAssign:
    """Provide a Emotion to the analyzed text."""
    def __init__(self):
        self.classifier = None
    def load_classifier(self):
        self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=0)
    def analyze(self,text):
        return self.classifier(text)
    def clear_cache(self):
        del self.classifier
        torch.cuda.empty_cache()