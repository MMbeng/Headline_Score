"""Score headlines using a pre-trained SVM and SentenceTransformer model."""

import sys
import os
import warnings
from datetime import datetime
from typing import List
import joblib
from sentence_transformers import SentenceTransformer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def load_headlines(file_path: str) -> List[str]:
    """Load headlines from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def score_headlines(headlines: List[str],
                    model: SentenceTransformer,
                    classifier) -> List[int]:
    """Encode headlines and return predictions from classifier."""
    embeddings = model.encode(headlines)
    return classifier.predict(embeddings)


def write_output_file(headlines: List[str],
                      predictions: List[int],
                      source: str) -> str:
    """Write predictions and headlines to an output file."""
    date_str = datetime.today().strftime("%Y_%m_%d")
    output_filename = f"headline_scores_{source}_{date_str}.txt"

    with open(output_filename, "w", encoding="utf-8") as file:
        for label, headline in zip(predictions, headlines):
            file.write(f"{label}\t{headline}\n")

    return output_filename


def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) != 3:
        print("Error: Please provide exactly two arguments.")
        print("Usage: python score_headlines.py <headline_file.txt> <source_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    source = sys.argv[2]

    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    headlines = load_headlines(input_file)
    if not headlines:
        print("Error: No headlines found in the file.")
        sys.exit(1)

    classifier = joblib.load("svm.joblib")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    predictions = score_headlines(headlines, embedding_model, classifier)
    output_file = write_output_file(headlines, predictions, source)

    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
