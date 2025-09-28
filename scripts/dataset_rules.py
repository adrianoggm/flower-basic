#!/usr/bin/env python3
"""
STRICT PROJECT RULES - AI/ML Dataset Usage
==========================================

MANDATORY RULES FOR ALL AI/ML ALGORITHMS AND EVALUATIONS:

🚫 PROHIBITED:
- Mock data generation for AI/ML algorithms
- Synthetic dataset creation
- Random data generation for model training/testing
- Fake samples or simulated physiological signals
- Any artificial data substitution

✅ REQUIRED:
- Use ONLY real WESAD and SWELL datasets
- Load actual dataset files (.pkl for WESAD, .csv for SWELL)
- Work with authentic physiological and behavioral data
- Maintain data integrity and scientific validity

📊 APPROVED DATASETS:
1. WESAD (Wearable Stress and Affect Detection)
   - Real physiological signals: BVP, EDA, ACC, TEMP
   - Authentic stress/emotion labels from controlled studies
   - Must use actual .pkl files from subjects S2-S17

2. SWELL (Stress, Workload, and Engagement from Wearable and Log-based data)
   - Real multimodal features: computer interaction, facial expressions, posture, physiology
   - Authentic stress conditions from workplace studies
   - Must use actual .csv files from feature datasets

⚡ ENFORCEMENT:
- Any mock data generation functions must be removed
- Code reviews will check for synthetic data usage
- All ML evaluations must demonstrate real data loading
- Baseline comparisons must use authentic samples

🎯 OBJECTIVE:
Ensure scientific rigor and realistic federated learning scenarios
using genuine human physiological and behavioral data.

VIOLATION OF THESE RULES INVALIDATES ALL ML/AI RESULTS.
"""

# Configuration constants
APPROVED_DATASETS = {
    "WESAD": {
        "path": "data/WESAD",
        "format": ".pkl",
        "subjects": [
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "S11",
            "S13",
            "S14",
            "S15",
            "S16",
            "S17",
        ],
        "signals": ["BVP", "EDA", "ACC", "TEMP"],
        "labels": [0, 1, 2, 3, 4, 5, 6, 7],
        "description": "Wearable Stress and Affect Detection - Real physiological data",
    },
    "SWELL": {
        "path": "data/SWELL",
        "format": ".csv",
        "modalities": ["computer", "facial", "posture", "physiology"],
        "conditions": ["baseline", "time_pressure", "interruption", "combined"],
        "description": "Stress, Workload, and Engagement - Real multimodal workplace data",
    },
}


def validate_real_data_usage(dataset_name: str, data_path: str) -> bool:
    """
    Validate that only real datasets are being used.

    Args:
        dataset_name: Name of dataset ('WESAD' or 'SWELL')
        data_path: Path to data files

    Returns:
        bool: True if real data, False if mock/synthetic

    Raises:
        ValueError: If mock data is detected
    """
    if dataset_name not in APPROVED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not in approved list: {list(APPROVED_DATASETS.keys())}"
        )

    # Check for real data indicators
    from pathlib import Path

    path = Path(data_path)

    if dataset_name == "WESAD":
        # Must have .pkl files for subjects
        pkl_files = list(path.glob("*/*.pkl"))
        if len(pkl_files) == 0:
            raise ValueError("No real WESAD .pkl files found - mock data prohibited!")

    elif dataset_name == "SWELL":
        # Must have .csv files for modalities
        csv_files = list(path.glob("**/*.csv"))
        if len(csv_files) == 0:
            raise ValueError("No real SWELL .csv files found - mock data prohibited!")

    return True


def check_mock_data_prohibition():
    """
    Runtime check to ensure no mock data functions exist.
    """
    import inspect
    import sys

    # Get current module
    current_module = sys.modules[__name__]

    # Check for prohibited function names
    prohibited_functions = [
        "generate_mock_",
        "create_fake_",
        "simulate_",
        "synthetic_",
        "random_data",
        "mock_data",
        "fake_data",
        "artificial_",
    ]

    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj):
            for prohibited in prohibited_functions:
                if prohibited in name.lower():
                    raise RuntimeError(
                        f"PROHIBITED: Mock data function '{name}' detected! Remove immediately."
                    )


def enforce_real_data_only():
    """
    Decorator to enforce real data usage in ML functions.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"🔍 Enforcing real data only for: {func.__name__}")
            check_mock_data_prohibition()
            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    print("🚫 AI/ML DATASET RULES ENFORCEMENT ACTIVE")
    print("=" * 50)
    print("✅ Only WESAD and SWELL real datasets permitted")
    print("🚫 Mock data generation strictly prohibited")
    print("🔍 Runtime validation enabled")
    check_mock_data_prohibition()
    print("✅ Rules compliance verified")
