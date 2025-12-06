"""
Comprehensive Benchmark Reporter

Generates human-readable reports from comprehensive benchmark results.
This module handles both console printing and file output of benchmark summaries.

Classes:
    comprehensive_benchmark_reporter: Generates and saves benchmark reports
"""

import constants
from constants import logging, os, json
from constants import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class numpy_json_encoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class comprehensive_benchmark_reporter:
    """
    Generates human-readable reports from comprehensive benchmark results.

    This class formats benchmark results for console output and creates
    text/JSON files with detailed summaries and statistics.
    """

    def __init__(self):
        """Initialize comprehensive benchmark reporter."""
        logger.info("comprehensive_benchmark_reporter initialized")

    def save_results(self, results: Dict, filename: str = None) -> None:
        """
        Save comprehensive benchmark results to JSON file.

        Args:
            results: Complete benchmark results dictionary
            filename: Output filename (default: comprehensive_benchmark_results.json)
        """
        if filename is None:
            filename = os.path.join(constants.PATH, "comprehensive_benchmark_results.json")

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, cls=numpy_json_encoder)

        logger.info(f"Results saved to {filename}")

    def print_summary(self, summary: Dict) -> None:
        """
        Print human-readable summary report to console.

        Args:
            summary: Summary statistics dictionary
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK SUMMARY REPORT")
        print("=" * 80)

        print(f"\nModel: {summary['metadata']['model_path']}")
        print(f"Runtime: {summary['metadata']['total_runtime_seconds']/60:.1f} minutes")
        print(f"Timestamp: {summary['metadata']['timestamp']}")

        # Training samples
        print("\n--- TRAINING SAMPLES (Exact Training Data) ---")
        self._print_category_summary(summary['training_samples_summary'])

        # Interpolation
        print("\n--- INTERPOLATION TESTS (Within Training Range) ---")
        self._print_category_summary(summary['interpolation_summary'])

        # Extrapolation
        print("\n--- EXTRAPOLATION TESTS (Beyond Training Range) ---")
        self._print_category_summary(summary['extrapolation_summary'])

        # Edge cases
        print("\n--- EDGE CASE TESTS (Boundary Conditions) ---")
        self._print_category_summary(summary['edge_cases_summary'])

        # Redundancy pairs
        print("\n--- REDUNDANCY PAIR TESTS (Scale-Equivalent Systems) ---")
        red_sum = summary['redundancy_summary']
        print(f"Total pairs:          {red_sum['total_pairs']}")
        print(f"Successful:           {red_sum['successful']}")
        print(f"Consistent pairs:     {red_sum['consistent_pairs']} ({red_sum['consistency_rate']*100:.1f}%)")
        if red_sum['k_magnitude_difference']['mean'] is not None:
            print(f"k-point difference:   {red_sum['k_magnitude_difference']['mean']:.6f} (mean), "
                  f"{red_sum['k_magnitude_difference']['max']:.6f} (max)")
            print(f"Velocity difference:  {red_sum['velocity_relative_difference_percent']['mean']:.2f}% (mean), "
                  f"{red_sum['velocity_relative_difference_percent']['max']:.2f}% (max)")

        print("\n" + "=" * 80)

    def save_summary_txt(self, summary: Dict, filename: str = None) -> None:
        """
        Save human-readable summary report to text file.

        Args:
            summary: Summary statistics dictionary
            filename: Output filename (default: comprehensive_benchmark_summary.txt)
        """
        if filename is None:
            filename = os.path.join(constants.PATH, "comprehensive_benchmark_summary.txt")

        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")

            f.write(f"\nModel: {summary['metadata']['model_path']}\n")
            f.write(f"Runtime: {summary['metadata']['total_runtime_seconds']/60:.1f} minutes\n")
            f.write(f"Timestamp: {summary['metadata']['timestamp']}\n")

            # Training samples
            f.write("\n--- TRAINING SAMPLES (Exact Training Data) ---\n")
            self._write_category_summary(f, summary['training_samples_summary'])

            # Interpolation
            f.write("\n--- INTERPOLATION TESTS (Within Training Range) ---\n")
            self._write_category_summary(f, summary['interpolation_summary'])

            # Extrapolation
            f.write("\n--- EXTRAPOLATION TESTS (Beyond Training Range) ---\n")
            self._write_category_summary(f, summary['extrapolation_summary'])

            # Edge cases
            f.write("\n--- EDGE CASE TESTS (Boundary Conditions) ---\n")
            self._write_category_summary(f, summary['edge_cases_summary'])

            # Redundancy pairs
            f.write("\n--- REDUNDANCY PAIR TESTS (Scale-Equivalent Systems) ---\n")
            red_sum = summary['redundancy_summary']
            f.write(f"Total pairs:          {red_sum['total_pairs']}\n")
            f.write(f"Successful:           {red_sum['successful']}\n")
            f.write(f"Consistent pairs:     {red_sum['consistent_pairs']} ({red_sum['consistency_rate']*100:.1f}%)\n")
            if red_sum['k_magnitude_difference']['mean'] is not None:
                f.write(f"k-point difference:   {red_sum['k_magnitude_difference']['mean']:.6f} (mean), "
                       f"{red_sum['k_magnitude_difference']['max']:.6f} (max)\n")
                f.write(f"Velocity difference:  {red_sum['velocity_relative_difference_percent']['mean']:.2f}% (mean), "
                       f"{red_sum['velocity_relative_difference_percent']['max']:.2f}% (max)\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Summary text file saved to {filename}")

    def _write_category_summary(self, f, cat_sum: Dict) -> None:
        """
        Write summary for a single category to file.

        Args:
            f: File handle
            cat_sum: Category summary dictionary
        """
        f.write(f"Total tests:          {cat_sum['total_tests']}\n")
        f.write(f"Successful:           {cat_sum['successful']} ({cat_sum['success_rate']*100:.1f}%)\n")

        # Only write timing/acceleration if available
        if 'acceleration_factor' in cat_sum:
            f.write(f"Acceleration:         {cat_sum['acceleration_factor']['mean']:.1f}x (mean), "
                   f"{cat_sum['acceleration_factor']['median']:.1f}x (median)\n")
        if 'nn_time_ms' in cat_sum:
            f.write(f"NN time:              {cat_sum['nn_time_ms']['mean']:.3f} ms (mean)\n")
        if 'physics_time_ms' in cat_sum:
            f.write(f"Physics time:         {cat_sum['physics_time_ms']['mean']:.1f} ms (mean)\n")

        if 'accuracy' in cat_sum and cat_sum['accuracy'].get('k_magnitude_error_mean') is not None:
            acc = cat_sum['accuracy']
            f.write(f"k-point error:        mean={acc['k_magnitude_error_mean']:.6f}, median={acc.get('k_magnitude_error_median', 0):.6f}\n")
            f.write(f"                      min={acc.get('k_magnitude_error_min', 0):.6f}, max={acc.get('k_magnitude_error_max', 0):.6f}\n")
            f.write(f"                      p25={acc.get('k_magnitude_error_p25', 0):.6f}, p75={acc.get('k_magnitude_error_p75', 0):.6f}\n")
            if acc.get('velocity_error_mean') is not None:
                f.write(f"Velocity error:       mean={acc['velocity_error_mean']:.4f}, median={acc.get('velocity_error_median', 0):.4f}\n")
                f.write(f"                      min={acc.get('velocity_error_min', 0):.4f}, max={acc.get('velocity_error_max', 0):.4f}\n")
                f.write(f"                      p25={acc.get('velocity_error_p25', 0):.4f}, p75={acc.get('velocity_error_p75', 0):.4f}\n")
            if acc.get('k_separation_mean') is not None:
                f.write(f"k-space separation:   {acc['k_separation_mean']:.6f} (mean, two-point)\n")

    def _print_category_summary(self, cat_sum: Dict) -> None:
        """
        Print summary for a single category to console.

        Args:
            cat_sum: Category summary dictionary
        """
        print(f"Total tests:          {cat_sum['total_tests']}")
        print(f"Successful:           {cat_sum['successful']} ({cat_sum['success_rate']*100:.1f}%)")

        # Only print timing/acceleration if available
        if 'acceleration_factor' in cat_sum:
            print(f"Acceleration:         {cat_sum['acceleration_factor']['mean']:.1f}x (mean), "
                  f"{cat_sum['acceleration_factor']['median']:.1f}x (median)")
        if 'nn_time_ms' in cat_sum:
            print(f"NN time:              {cat_sum['nn_time_ms']['mean']:.3f} ms (mean)")
        if 'physics_time_ms' in cat_sum:
            print(f"Physics time:         {cat_sum['physics_time_ms']['mean']:.1f} ms (mean)")

        if 'accuracy' in cat_sum and cat_sum['accuracy'].get('k_magnitude_error_mean') is not None:
            acc = cat_sum['accuracy']
            print(f"k-point error:        mean={acc['k_magnitude_error_mean']:.6f}, median={acc.get('k_magnitude_error_median', 0):.6f}")
            print(f"                      min={acc.get('k_magnitude_error_min', 0):.6f}, max={acc.get('k_magnitude_error_max', 0):.6f}")
            print(f"                      p25={acc.get('k_magnitude_error_p25', 0):.6f}, p75={acc.get('k_magnitude_error_p75', 0):.6f}")
            if acc.get('velocity_error_mean') is not None:
                print(f"Velocity error:       mean={acc['velocity_error_mean']:.4f}, median={acc.get('velocity_error_median', 0):.4f}")
                print(f"                      min={acc.get('velocity_error_min', 0):.4f}, max={acc.get('velocity_error_max', 0):.4f}")
                print(f"                      p25={acc.get('velocity_error_p25', 0):.4f}, p75={acc.get('velocity_error_p75', 0):.4f}")
            if acc.get('k_separation_mean') is not None:
                print(f"k-space separation:   {acc['k_separation_mean']:.6f} (mean, two-point)")
