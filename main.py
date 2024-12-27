import argparse
import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from src.DFHandler import DFHandler
from src.Visualizer import Visualizer
from src.EmoAssign import EmoAssign
from src.SemTagger import SemTagger


def check_cuda():
    """Checks CUDA availability and PyTorch version."""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")


def analyze(args):
    """Performs emotion and semantic analysis."""
    handler = DFHandler()
    emo = EmoAssign()
    sem_tag = SemTagger()

    # Collect and parse the text
    speech_path = args.input
    phrase_list = handler.txt_parser(speech_path)
    df = handler.list_to_df(phrase_list)

    # Emotion labelling
    phrase_list = df["Phrase"].tolist()
    emo.load_classifier()
    emotions = emo.classifier(phrase_list)
    emo.clear_cache()
    df = handler.update_emo_df(df, emotions)

    # Semantic evaluation
    sem_tag.load_model()
    semantic_output = sem_tag.predict_sentiment(phrase_list)
    sem_tag.clear_cache()
    df = handler.update_sem_df(df, semantic_output)

    # Save the output
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output}")


def visualize(args):
    """Visualizes output data."""
    # Load the dataset
    df = pd.read_csv(args.input)

    vis = Visualizer()
    emotion_columns = ['Neutral', 'Joy', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Sadness']

    # Choose visualization type
    if args.type == "grouped":
        plt = vis.grouped_bar_chart(df, emotion_columns, start=args.start, end=args.end)
    elif args.type == "stacked":
        plt = vis.stacked_bar_chart(df)
    elif args.type == "semantic":
        plt = vis.create_sem_dynamics(df)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize emotions and semantics in text.")

    # Add subparsers
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Subparser for CUDA check
    cuda_parser = subparsers.add_parser("check-cuda", help="Check CUDA and PyTorch installation.")
    cuda_parser.set_defaults(func=check_cuda)

    # Subparser for analysis
    analyze_parser = subparsers.add_parser("analyze", help="Analyze emotions and semantics.")
    analyze_parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input text file.")
    analyze_parser.add_argument("-o", "--output", type=str, default="output.csv", help="Path to save the output csv file.")
    analyze_parser.set_defaults(func=analyze)

    # Subparser for visualization
    visualize_parser = subparsers.add_parser("visualize", help="Visualize emotion data.")
    visualize_parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input dataset (CSV).")
    visualize_parser.add_argument("-t", "--type", type=str, choices=["grouped", "stacked", "semantic"], required=True,
                                   help="Type of visualization: grouped, stacked, semantic.")
    visualize_parser.add_argument("-s", "--start", type=int, default=0, help="Start index for grouped bar chart.")
    visualize_parser.add_argument("-e", "--end", type=int, default=10, help="End index for grouped bar chart.")
    visualize_parser.set_defaults(func=visualize)

    # Parse arguments and invoke the appropriate function
    args = parser.parse_args()
    args.func(args)
