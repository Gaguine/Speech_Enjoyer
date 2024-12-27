from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class SemTagger:
    """Provide a Semantic tag to the analysed text"""
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        model_name = "tabularisai/multilingual-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict_sentiment(self,text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        return torch.argmax(probabilities, dim=-1).tolist() #[sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]
    def clear_cache(self):
        del self.tokenizer
        torch.cuda.empty_cache()