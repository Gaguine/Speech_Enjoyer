import pandas as pd

class DFHandler:
    def __init__(self):
        self.df = None
    def txt_parser(self,path: str) -> list[str]:
        """Parse and clean the text."""
        with open(path, "r", encoding="UTF-8") as file:
            stripped_phrases = [phrase.strip() for phrase in file.readlines()]
            return stripped_phrases
    def list_to_df(self,phrases:list[str])->pd.DataFrame:
        df = pd.DataFrame({
            "Phrase":phrases,
            "Neutral": None,
            "Joy":None,
            "Anger":None,
            "Fear":None,
            "Surprise":None,
            "Disgust":None,
            "Sadness":None,
            "Semantic_Score":None
        })
        return df
    def update_emo_df(self,df: pd.DataFrame,emotions:list[list[dict]]) -> pd.DataFrame:
        """The order of the labels stays the same:
        1. NEUTRAL
        2. JOY
        3. ANGER
        5. FEAR
        6. SURPRISE
        7. DISGUST
        8. SADNESS
        If the value is closer to 1, it means that it is pretty probable that we can apply that particular label.
        """

        neutral_list = []
        joy_list = []
        anger_list = []
        fear_list = []
        surprise_list = []
        disgust_list = []
        sadness_list = []

        for ph_emotion in emotions:
            neutral_list.append(ph_emotion[0]['score'])
            joy_list.append(ph_emotion[1]['score'])
            anger_list.append(ph_emotion[2]['score'])
            fear_list.append(ph_emotion[3]['score'])
            surprise_list.append(ph_emotion[4]['score'])
            disgust_list.append(ph_emotion[5]['score'])
            sadness_list.append(ph_emotion[6]['score'])

        df["Neutral"] = neutral_list
        df["Joy"] = joy_list
        df["Anger"] = anger_list
        df["Fear"] = fear_list
        df["Surprise"] = surprise_list
        df["Disgust"] = disgust_list
        df["Sadness"] = sadness_list
        return df
    def update_sem_df(self,df:pd.DataFrame,semantic_output:list[int]) -> pd.DataFrame:
        df["Semantic_Score"] = semantic_output
        return df