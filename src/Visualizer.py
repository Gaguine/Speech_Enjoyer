import pandas as pd
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self,df=None):
        self.dataset = df
    def create_sem_dynamics(self,df:pd.DataFrame) -> plt:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Semantic_Score'], marker='o', linestyle='-', label='Semantic Score Dynamics')

        # Adding titles and labels
        plt.title('Semantic Tag Dynamics Throughout the Speech', fontsize=14)
        plt.xlabel('Phrase Index', fontsize=12)
        plt.ylabel('Semantic Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        return plt
    def stacked_bar_chart(self, df: pd.DataFrame) -> plt:
        # Emotion columns
        emotion_columns = ['Neutral', 'Joy', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Sadness']

        # Calculate the total sum for each emotion
        total_scores = df[emotion_columns].sum()

        # Create the stacked bar chart
        plt.figure(figsize=(12, 6))
        total_scores.plot(kind='bar', stacked=True,
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])

        # Add titles and labels
        plt.title('Overall Emotion Score Comparison (Stacked)', fontsize=16)
        plt.xlabel('Emotions', fontsize=14)
        plt.ylabel('Total Scores', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)

        return plt
    def grouped_bar_chart(self, df: pd.DataFrame, emotion_columns: list, start: int = 0, end: int = 10) -> plt:
        """
        Creates a grouped bar chart to compare emotion scores for a specific range of phrases.

        Args:
        - df (pd.DataFrame): The DataFrame containing emotion scores.
        - emotion_columns (list): List of emotion columns to visualize.
        - start (int): Starting index for the phrase range.
        - end (int): Ending index for the phrase range.
        """
        # Slice the DataFrame for the specified range
        phrase_range = df[emotion_columns].iloc[start:end]

        # Plot the grouped bar chart
        plt.figure(figsize=(14, 8))
        phrase_range.plot(kind='bar', figsize=(14, 8), width=0.8)

        # Add titles and labels
        plt.title(f'Emotion Score Comparison for Phrases {start} to {end - 1}', fontsize=16)
        plt.xlabel('Phrase Index', fontsize=14)
        plt.ylabel('Emotion Scores', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(ticks=range(len(phrase_range.index)), labels=phrase_range.index, rotation=0, fontsize=10)
        plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

        return plt