"""
Preprocess training data to group multiple Dirac points by parameter set.

This script:
1. Loads the training CSV
2. Groups all Dirac points that belong to the same parameter set
3. Removes duplicate Dirac points (same k_x, k_y, velocity)
4. Saves in efficient format for minimum distance loss training
"""

import csv
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import constants
from constants import logging, os

logger = logging.getLogger(__name__)

def preprocess_training_data(input_csv: str = 'Training_data/dirac_training_data.csv',
                             output_file: str = 'Training_data/grouped_training_data.pkl') -> None:
    """
    Preprocess training data by grouping Dirac points by parameter set.

    Automatically loads both dirac_training_data.csv and dirac_training_data_old.csv
    if they exist in the Training_data directory.

    Args:
        input_csv: Path to original training CSV (or directory to search for CSVs)
        output_file: Path to save preprocessed data (pickle format)
    """

    # Dictionary: parameter_tuple -> list of (k_x, k_y, velocity) tuples
    param_to_dirac_points: Dict[Tuple, List[Tuple[float, float, float]]] = defaultdict(list)

    # Determine CSV files to load
    data_dir = os.path.dirname(input_csv)
    if not data_dir:
        data_dir = 'Training_data'

    csv_files = [
        os.path.join(data_dir, 'dirac_training_data.csv'),
        os.path.join(data_dir, 'dirac_training_data_old.csv')
    ]

    # Filter to only existing files
    csv_files = [f for f in csv_files if os.path.exists(f)]

    if not csv_files:
        logger.error(f"No training CSV files found in {data_dir}")
        logger.error(f"Checked for: {os.path.join(data_dir, 'dirac_training_data.csv')}")
        logger.error(f"Checked for: {os.path.join(data_dir, 'dirac_training_data_old.csv')}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"No training data found in {data_dir}")

    logger.info(f"Found {len(csv_files)} training CSV file(s) to process:")
    for csv_file in csv_files:
        logger.info(f"  - {csv_file}")

    # Load and group data from all CSV files
    total_rows = 0
    for csv_file in csv_files:
        logger.info(f"\nLoading training data from {csv_file}...")
        file_rows = 0

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                file_rows += 1

                # Extract 6 parameters
                params = (
                    int(row['a']),
                    int(row['b']),
                    float(row['interlayer_dist_threshold']),
                    float(row['intralayer_dist_threshold']),
                    float(row['inter_graph_weight']),
                    float(row['intra_graph_weight'])
                )

                # Extract Dirac point
                k_x = float(row['target_k_x'])
                k_y = float(row['target_k_y'])
                velocity = float(row['Dirac_velocity'])

                param_to_dirac_points[params].append((k_x, k_y, velocity))

        logger.info(f"  Loaded {file_rows:,} samples from this file")

    logger.info(f"\nTotal loaded: {total_rows:,} training samples across all files")
    logger.info(f"Found {len(param_to_dirac_points):,} unique parameter sets")

    # Remove duplicate Dirac points for each parameter set
    logger.info("Removing duplicate Dirac points...")

    deduplicated_data: Dict[Tuple, List[Tuple[float, float, float]]] = {}
    total_points_before = 0
    total_points_after = 0

    for params, points in param_to_dirac_points.items():
        total_points_before += len(points)

        # Remove duplicates (within tolerance)
        unique_points = []
        for p in points:
            is_duplicate = False
            for up in unique_points:
                # Check if points are within 1e-6 tolerance
                if (abs(p[0] - up[0]) < 1e-6 and
                    abs(p[1] - up[1]) < 1e-6 and
                    abs(p[2] - up[2]) < 1e-6):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(p)

        deduplicated_data[params] = unique_points
        total_points_after += len(unique_points)

    logger.info(f"Reduced from {total_points_before:,} to {total_points_after:,} unique Dirac points")
    logger.info(f"Removed {total_points_before - total_points_after:,} duplicates ({(total_points_before - total_points_after)/total_points_before*100:.1f}%)")

    # Analyze distribution
    num_points_distribution = defaultdict(int)
    for params, points in deduplicated_data.items():
        num_points_distribution[len(points)] += 1

    logger.info("\nDistribution of Dirac points per parameter set:")
    for num_points in sorted(num_points_distribution.keys()):
        count = num_points_distribution[num_points]
        pct = count / len(deduplicated_data) * 100
        logger.info(f"  {num_points:2d} points: {count:5d} parameter sets ({pct:5.1f}%)")

    # Convert to format suitable for training
    logger.info("\nConverting to training format with input transformation...")

    training_data = []
    for params, dirac_points in deduplicated_data.items():
        # Extract raw parameters
        a, b, interlayer_thresh, intralayer_thresh, inter_weight, intra_weight = params

        # CRITICAL: Apply input transformation (a, b) → (1/a, 1/b) for numerical stability
        # Network inputs must be in similar scale (all in [0,1] range)
        # Without this transformation: a,b ∈ [2,333] while other params ∈ [0.15,1.8]
        # This would cause massive gradient issues and prevent learning
        inv_a = 1.0 / float(a)
        inv_b = 1.0 / float(b)

        # Create transformed parameter array with clear naming
        # Order: [1/a, 1/b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight]
        transformed_params = np.array([
            inv_a, inv_b,
            interlayer_thresh, intralayer_thresh,
            inter_weight, intra_weight
        ], dtype=np.float32)

        # Dirac points remain unchanged: [k_x, k_y, velocity]
        dirac_points_array = np.array(dirac_points, dtype=np.float32)  # Shape: (num_points, 3)

        training_data.append({
            'parameters_transformed': transformed_params,  # Clear name: these are TRANSFORMED inputs
            'parameters_raw': np.array(params, dtype=np.float32),  # Keep original for reference
            'dirac_points': dirac_points_array,
            'num_points': len(dirac_points)
        })

    # Save preprocessed data
    logger.info(f"\nSaving preprocessed data to {output_file}...")

    output_data = {
        'training_samples': training_data,
        'num_parameter_sets': len(training_data),
        'total_dirac_points': total_points_after,
        'preprocessing_info': {
            'original_samples': total_rows,
            'unique_parameter_sets': len(param_to_dirac_points),
            'duplicates_removed': total_points_before - total_points_after,
            'distribution': dict(num_points_distribution)
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Preprocessing complete!")
    logger.info(f"\nSummary:")
    logger.info(f"  Input: {total_rows:,} samples from CSV")
    logger.info(f"  Output: {len(training_data):,} parameter sets with {total_points_after:,} unique Dirac points")
    logger.info(f"  Average: {total_points_after/len(training_data):.2f} Dirac points per parameter set")
    logger.info(f"  File saved: {output_file}")

    return output_data


def load_preprocessed_data(input_file: str = 'Training_data/grouped_training_data.pkl') -> Dict:
    """
    Load preprocessed training data.

    Args:
        input_file: Path to preprocessed data file

    Returns:
        Dict containing training_samples and metadata
    """
    logger.info(f"Loading preprocessed data from {input_file}...")

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"Loaded {data['num_parameter_sets']:,} parameter sets")
    logger.info(f"Total {data['total_dirac_points']:,} unique Dirac points")

    return data


if __name__ == "__main__":
    # Run preprocessing
    preprocess_training_data()

    # Test loading
    print("\n" + "="*80)
    print("Testing data loading...")
    print("="*80 + "\n")

    data = load_preprocessed_data()

    # Show a few examples
    print("First 5 parameter sets:")
    for i, sample in enumerate(data['training_samples'][:5], 1):
        params_raw = sample['parameters_raw']
        params_transformed = sample['parameters_transformed']
        num_points = sample['num_points']
        dirac_points = sample['dirac_points']

        print(f"\n{i}. Raw Parameters: a={int(params_raw[0])}, b={int(params_raw[1])}, "
              f"thresholds=({params_raw[2]:.2f}, {params_raw[3]:.2f}), "
              f"weights=({params_raw[4]:.2f}, {params_raw[5]:.2f})")
        print(f"   Transformed Inputs: [1/a={params_transformed[0]:.4f}, 1/b={params_transformed[1]:.4f}, "
              f"{params_transformed[2]:.2f}, {params_transformed[3]:.2f}, "
              f"{params_transformed[4]:.2f}, {params_transformed[5]:.2f}]")
        print(f"   {num_points} Dirac points:")

        for j, (k_x, k_y, v) in enumerate(dirac_points, 1):
            print(f"     {j}. k=({k_x:7.4f}, {k_y:7.4f}), v={v:.4f}")
