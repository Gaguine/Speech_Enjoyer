import pandas as pd
from src.Visualizer import Visualizer
from src.DFHandler import DFHandler
from src.SemTagger import SemTagger
from src.EmoAssign import EmoAssign
import torch
import os
import time


"""
This should be a straightforward project.

1. The text should be already cleaned, yet to be parsed.
2. create an inference, where the LLM model analyses each phrase.
3. Store the gained information(probably a semantic value)
4. Plot the data to see the emotion dynamics, using matplot lib
"""
def check_cuda():
    print(torch.cuda.is_available())
    print(torch.__version__)  # Prints the PyTorch version
    print(torch.version.cuda)
start_timer = time.time()

# Create objects
handler = DFHandler()
emo = EmoAssign()
sem_tag = SemTagger()

#Collect and parse the text
speech_path = os.path.join("speech/Elon Musk Speech")
phrase_list = handler.txt_parser(speech_path)
df = handler.list_to_df(phrase_list)

# Analyse phrases and update the values
"""Emotion Labelling"""
phrase_list = df["Phrase"].tolist()
emo.load_classifier()
emotions = emo.classifier(phrase_list)
emo.clear_cache()
df = handler.update_emo_df(df,emotions)
"""Semantic Evaluation"""
sem_tag.load_model()
semantic_output = sem_tag.predict_sentiment(phrase_list)
sem_tag.clear_cache()
df = handler.update_sem_df(df,semantic_output)

"""Create Dataset Output"""
df.to_excel("output.xlsx")

"""Visualisation"""
df = pd.read_csv("output.csv")
vis  = Visualizer()
emotion_columns = ['Neutral', 'Joy', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Sadness']
plt = vis.grouped_bar_chart(df, emotion_columns, start=0, end=5)

plt.tight_layout()
plt.show()
end_time = time.time()
print(f"Script runtime: {end_time - start_timer:.2f} seconds")
