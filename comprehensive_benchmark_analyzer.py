"""
Comprehensive Benchmark Analyzer

Analyzes benchmark results and generates summary statistics for comprehensive benchmarks.
This module extracts metrics, analyzes redundancy consistency, and summarizes test categories.

Classes:
    comprehensive_benchmark_analyzer: Analyzes benchmark results and computes statistics
"""

import constants
from constants import np, logging
from constants import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)


class comprehensive_benchmark_analyzer:
    """
    Analyzes comprehensive benchmark results and generates summary statistics.

    This class processes raw benchmark results to extract key metrics, analyze
    redundancy pair consistency, and generate statistical summaries for each
    test category.
    """

    def __init__(self):
        """Initialize comprehensive benchmark analyzer."""
        logger.info("comprehensive_benchmark_analyzer initialized")

    def extract_key_metrics(self, benchmark_result: Dict) -> Dict:
        """
        Extract key metrics from a single benchmark result.

        Args:
            benchmark_result: Raw benchmark result dictionary

        Returns:
            Dict containing extracted metrics (acceleration, timing, predictions, errors)
        """
        if 'error' in benchmark_result:
            return {'error': benchmark_result['error']}

        metrics = {
            'acceleration_factor': benchmark_result['acceleration_factor'],
            'nn_time_ms': benchmark_result['average_nn_time_ms'],
            'physics_time_ms': benchmark_result['average_physics_time_ms']
        }

        # Add accuracy if available
        if 'accuracy_comparison' in benchmark_result and 'error' not in benchmark_result['accuracy_comparison']:
            accuracy = benchmark_result['accuracy_comparison']

            # Get detailed results if available
            if 'comparison_details' in accuracy and len(accuracy['comparison_details']) > 0:
                detail = accuracy['comparison_details'][0]
                metrics['prediction'] = {
                    'k_x': detail.get('nn_k_x'),
                    'k_y': detail.get('nn_k_y'),
                    'velocity': detail.get('nn_velocity')
                }
                metrics['physics_result'] = {
                    'k_x': detail.get('physics_k_x'),
                    'k_y': detail.get('physics_k_y'),
                    'velocity': detail.get('physics_velocity')
                }
                metrics['errors'] = {
                    'k_x_error': detail.get('k_x_error'),
                    'k_y_error': detail.get('k_y_error'),
                    'k_magnitude_error': detail.get('k_magnitude_error'),
                    'velocity_error': detail.get('velocity_error'),
                    'velocity_rel_error_percent': detail.get('velocity_relative_error_percent')
                }

        return metrics

    def analyze_redundancy_consistency(self, result_1: Dict, result_2: Dict,
                                      params_1: List, params_2: List) -> Dict:
        """
        Analyze consistency between redundancy pair results.

        Since redundancy pairs represent the same physical system (scale-equivalent),
        their predictions should be very similar.

        Args:
            result_1: First system's benchmark result
            result_2: Second system's benchmark result
            params_1: First system's parameters [a, b, ...]
            params_2: Second system's parameters [a, b, ...]

        Returns:
            Dict containing consistency metrics and analysis
        """
        if 'error' in result_1 or 'error' in result_2:
            return {'error': 'One or both tests failed'}

        # Extract predictions
        metrics_1 = self.extract_key_metrics(result_1)
        metrics_2 = self.extract_key_metrics(result_2)

        if 'prediction' not in metrics_1 or 'prediction' not in metrics_2:
            return {'error': 'Predictions not available'}

        pred_1 = metrics_1['prediction']
        pred_2 = metrics_2['prediction']

        # Calculate differences (should be near zero for perfect consistency)
        k_x_diff = abs(pred_1['k_x'] - pred_2['k_x'])
        k_y_diff = abs(pred_1['k_y'] - pred_2['k_y'])
        k_magnitude_diff = np.sqrt((pred_1['k_x'] - pred_2['k_x'])**2 +
                                   (pred_1['k_y'] - pred_2['k_y'])**2)
        velocity_diff = abs(pred_1['velocity'] - pred_2['velocity'])
        velocity_rel_diff = velocity_diff / abs(pred_1['velocity']) if abs(pred_1['velocity']) > 1e-10 else 0

        consistency = {
            'a_b_ratio': float(params_1[0] / params_1[1]),  # Should be same for both
            'system_1_ab': [float(params_1[0]), float(params_1[1])],
            'system_2_ab': [float(params_2[0]), float(params_2[1])],
            'k_x_difference': float(k_x_diff),
            'k_y_difference': float(k_y_diff),
            'k_magnitude_difference': float(k_magnitude_diff),
            'velocity_difference': float(velocity_diff),
            'velocity_relative_difference_percent': float(velocity_rel_diff * 100),
            'is_consistent': bool(k_magnitude_diff < constants.BENCHMARK_CONSISTENCY_K_ERROR_THRESHOLD and
                                  velocity_rel_diff < constants.BENCHMARK_CONSISTENCY_VEL_ERROR_THRESHOLD)
        }

        return consistency

    def generate_summary_report(self, results: Dict) -> Dict[str, Any]:
        """
        Generate summary statistics from benchmark results.

        Args:
            results: Complete benchmark results dictionary

        Returns:
            Dict containing summary statistics for all test categories
        """
        summary = {
            'metadata': results['metadata'],
            'training_samples_summary': self.summarize_category(results.get('training_samples', [])),
            'interpolation_summary': self.summarize_category(results['interpolation']),
            'extrapolation_summary': self.summarize_category(results['extrapolation']),
            'edge_cases_summary': self.summarize_category(results['edge_cases']),
            'redundancy_summary': self.summarize_redundancy_pairs(results['redundancy_pairs'])
        }

        return summary

    def summarize_category(self, category_results: List[Dict]) -> Dict:
        """
        Generate summary statistics for a test category.

        Args:
            category_results: List of test results for a category

        Returns:
            Dict containing statistical summary of the category
        """
        successful = [r for r in category_results if r.get('success', False)]
        failed = [r for r in category_results if not r.get('success', False)]

        if not successful:
            return {
                'total_tests': len(category_results),
                'successful': 0,
                'failed': len(failed),
                'success_rate': 0.0
            }

        # Collect metrics (filter out results that don't have these fields)
        acceleration_factors = [r['acceleration_factor'] for r in successful if 'acceleration_factor' in r]
        nn_times = [r['nn_time_ms'] for r in successful if 'nn_time_ms' in r]
        physics_times = [r['physics_time_ms'] for r in successful if 'physics_time_ms' in r]

        # If no successful results with metrics, return early
        if not acceleration_factors:
            return {
                'total_tests': len(category_results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(category_results) if category_results else 0.0
            }

        summary = {
            'total_tests': len(category_results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(category_results),
            'acceleration_factor': {
                'mean': float(np.mean(acceleration_factors)),
                'median': float(np.median(acceleration_factors)),
                'std': float(np.std(acceleration_factors)),
                'min': float(np.min(acceleration_factors)),
                'max': float(np.max(acceleration_factors))
            },
            'nn_time_ms': {
                'mean': float(np.mean(nn_times)),
                'median': float(np.median(nn_times))
            },
            'physics_time_ms': {
                'mean': float(np.mean(physics_times)),
                'median': float(np.median(physics_times))
            }
        }

        # Accuracy metrics if available
        k_mag_errors = [r['accuracy']['k_magnitude_error_mean']
                       for r in successful if 'accuracy' in r and r['accuracy'].get('k_magnitude_error_mean') is not None]
        vel_errors = [r['accuracy']['velocity_error_mean']
                     for r in successful if 'accuracy' in r and r['accuracy'].get('velocity_error_mean') is not None]

        if k_mag_errors:
            summary['accuracy'] = {
                'k_magnitude_error_mean': float(np.mean(k_mag_errors)),
                'k_magnitude_error_median': float(np.median(k_mag_errors)),
                'k_magnitude_error_min': float(np.min(k_mag_errors)),
                'k_magnitude_error_max': float(np.max(k_mag_errors)),
                'k_magnitude_error_std': float(np.std(k_mag_errors)),
                'k_magnitude_error_p25': float(np.percentile(k_mag_errors, 25)),
                'k_magnitude_error_p75': float(np.percentile(k_mag_errors, 75)),
                'velocity_error_mean': float(np.mean(vel_errors)) if vel_errors else None,
                'velocity_error_median': float(np.median(vel_errors)) if vel_errors else None,
                'velocity_error_min': float(np.min(vel_errors)) if vel_errors else None,
                'velocity_error_max': float(np.max(vel_errors)) if vel_errors else None,
                'velocity_error_std': float(np.std(vel_errors)) if vel_errors else None,
                'velocity_error_p25': float(np.percentile(vel_errors, 25)) if vel_errors else None,
                'velocity_error_p75': float(np.percentile(vel_errors, 75)) if vel_errors else None
            }

        # K-space separation for two-point predictions
        k_sep_means = [r['accuracy']['k_separation_mean']
                      for r in successful if 'accuracy' in r and r['accuracy'].get('k_separation_mean') is not None]
        if k_sep_means:
            if 'accuracy' not in summary:
                summary['accuracy'] = {}
            summary['accuracy']['k_separation_mean'] = float(np.mean(k_sep_means))
            summary['accuracy']['k_separation_median'] = float(np.median(k_sep_means))

        return summary

    def summarize_redundancy_pairs(self, redundancy_results: List[Dict]) -> Dict:
        """
        Summarize redundancy pair test results.

        Args:
            redundancy_results: List of redundancy pair test results

        Returns:
            Dict containing consistency statistics for redundancy pairs
        """
        successful = [r for r in redundancy_results if r.get('success', False)]

        if not successful:
            return {
                'total_pairs': len(redundancy_results),
                'successful': 0,
                'failed': len(redundancy_results)
            }

        # Analyze consistency for each pair
        analyzed_pairs = []
        for r in successful:
            if 'result_1' in r and 'result_2' in r:
                consistency = self.analyze_redundancy_consistency(
                    r['result_1'], r['result_2'],
                    r['params_1'], r['params_2']
                )
                r['consistency'] = consistency
                analyzed_pairs.append(r)

        # Collect consistency metrics
        k_diffs = [r['consistency']['k_magnitude_difference'] for r in analyzed_pairs
                  if 'consistency' in r and 'k_magnitude_difference' in r['consistency']]
        vel_diffs = [r['consistency']['velocity_relative_difference_percent'] for r in analyzed_pairs
                    if 'consistency' in r and 'velocity_relative_difference_percent' in r['consistency']]
        consistent_count = sum(1 for r in analyzed_pairs
                              if 'consistency' in r and r['consistency'].get('is_consistent', False))

        summary = {
            'total_pairs': len(redundancy_results),
            'successful': len(successful),
            'failed': len(redundancy_results) - len(successful),
            'consistent_pairs': consistent_count,
            'consistency_rate': float(consistent_count / len(successful)) if successful else 0.0,
            'k_magnitude_difference': {
                'mean': float(np.mean(k_diffs)) if k_diffs else None,
                'median': float(np.median(k_diffs)) if k_diffs else None,
                'max': float(np.max(k_diffs)) if k_diffs else None
            },
            'velocity_relative_difference_percent': {
                'mean': float(np.mean(vel_diffs)) if vel_diffs else None,
                'median': float(np.median(vel_diffs)) if vel_diffs else None,
                'max': float(np.max(vel_diffs)) if vel_diffs else None
            }
        }

        return summary
