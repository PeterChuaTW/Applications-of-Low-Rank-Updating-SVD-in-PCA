from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    # Data
    data_dir: str = "data/ORL_Faces"
    output_dir: str = "output"

    # PCA
    batch_size: int = 10

    # Rank selection
    energy_threshold: float = 0.95
    thresholds: tuple = (0.90, 0.95, 0.99)

    # Diagnostics
    max_lags: int = 50
    alpha: float = 0.05
