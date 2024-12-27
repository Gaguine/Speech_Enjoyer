import pandas as pd

class DFHandler:
    def __init__(self):
        self.df = None
    def txt_parser(self,path: str) -> list[str]:
        with open(path, "r", encoding="UTF-8") as file:
            return file.readlines()
    def list_to_df(self,phrases:list[str])->pd.DataFrame:
        df = pd.DataFrame({"Phrase":phrases,"Emotion":None,"Semantic_Score":None})
        return df