"""
Entry point for running the full PCA analysis pipeline.
"""

from src.experiments.config import AnalysisConfig
from src.experiments.pipeline import run_full_analysis


def main():
    config = AnalysisConfig(
        data_dir="data/ORL_Faces",
        output_dir="output",
        batch_size=10,
        energy_threshold=0.95
    )
    run_full_analysis(config)


if __name__ == "__main__":
    main()
