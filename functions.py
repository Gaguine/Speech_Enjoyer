import os


def txt_parser(path:str)->list[str]:
    with open(speech_path, "r",encoding="UTF-8") as file:
        return file.readlines()
def sentiment_analysis(text:str):
    """This function should return a semantic score. Preferably a float."""







"""Testing Ground"""
speech_path = "speech/Elon Musk Speech"
phrase_list = txt_parser(speech_path)
