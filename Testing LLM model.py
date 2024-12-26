from functions import speech_path, txt_parser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# for text, sentiment in zip(phrase_list[:5], predict_sentiment(phrase_list[:5])):
#     print(f"Text: {text}\nSentiment: {sentiment}\n")


class Umnik:
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
class RidlsRofler:



"""Testing Ground"""
umnik = Umnik()
phrase_list = txt_parser(speech_path)

for phrase in phrase_list[:7]:
    umnik.load_model()
    prediction = umnik.predict_sentiment(phrase)
    print(phrase, type(prediction),prediction)
