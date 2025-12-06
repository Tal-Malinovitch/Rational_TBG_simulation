"""
Performance benchmarking and acceleration analysis for Dirac point neural networks.

This module provides comprehensive benchmarking capabilities for measuring
neural network performance versus physics computations, following project
coding standards and integrating with existing utilities.

REFACTORED: Uses dirac_network_accuracy and dirac_network_report modules for
accuracy analysis and report generation.

Classes:
    dirac_network_benchmark: Performance analysis and acceleration factor measurement
"""

import constants
from constants import np, time, logging, os
from constants import List, Dict, Tuple, Optional, Union, Any
from TBG import Dirac_analysis
from neural_network_base import neural_network
from stats import statistics
from utils import validate_ab
from Generate_training_data import Find_Dirac_point

# Import refactored modules
from dirac_network_accuracy import dirac_network_accuracy_analyzer
from dirac_network_report import dirac_network_report_generator

# Configure logging
logger = logging.getLogger(__name__)

# Default benchmark configuration
default_benchmark_config = {
    'num_iterations': 10,
    'warmup_iterations': 3,
    'statistical_confidence': 0.95,
    'timing_precision': 1e-6  # microsecond precision
}


class dirac_network_benchmark:
    """
    Provides comprehensive benchmarking for Dirac point neural networks.

    This class measures and compares performance between neural network predictions
    and physics-based computations, calculating acceleration factors and providing
    detailed statistical analysis using project utilities.

    Attributes:
        benchmark_config (dict): Benchmark configuration parameters
        stats (statistics): Statistics tracking using project utility
        nn_prediction_times (List[float]): Recorded NN prediction times
        physics_computation_times (List[float]): Recorded physics computation times
        current_network (Optional[neural_network]): Network being benchmarked
        current_network_builder (Optional): Associated network builder
        accuracy_analyzer (dirac_network_accuracy_analyzer): Accuracy analysis
        report_generator (dirac_network_report_generator): Report generation
    """

    def __init__(self, benchmark_config: Optional[dict] = None) -> None:
        """
        Initialize the Dirac network benchmark system.

        Args:
            benchmark_config (dict, optional): Benchmark configuration parameters.
                Uses default_benchmark_config if not provided.

        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        try:
            self.benchmark_config = benchmark_config or default_benchmark_config.copy()
            self._validate_benchmark_config()

            # Initialize statistics using project utility
            self.stats = statistics()

            # Timing data storage
            self.nn_prediction_times: List[float] = []
            self.physics_computation_times: List[float] = []

            # Detailed results storage for accuracy comparison
            self.detailed_benchmark_results: List[Dict[str, Any]] = []

            # Network references
            self.current_network: Optional[neural_network] = None
            self.current_network_builder: Optional = None

            # Initialize refactored modules
            self.accuracy_analyzer = dirac_network_accuracy_analyzer()
            self.report_generator = dirac_network_report_generator()

            logger.info(f"dirac_network_benchmark initialized with config: {self.benchmark_config}")

        except Exception as e:
            logger.error(f"Failed to initialize dirac_network_benchmark: {str(e)}")
            raise constants.physics_parameter_error(f"Benchmark initialization failed: {str(e)}")

    def _validate_benchmark_config(self) -> None:
        """
        Validate benchmark configuration parameters.

        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        required_keys = ['num_iterations', 'warmup_iterations']

        for key in required_keys:
            if key not in self.benchmark_config:
                raise constants.physics_parameter_error(f"Missing required benchmark config key: {key}")

        # Validate positive integer values
        for key in required_keys:
            value = self.benchmark_config[key]
            if not isinstance(value, int) or value <= 0:
                raise constants.physics_parameter_error(f"Benchmark config {key} must be positive integer, got {value}")

        # Warmup should not exceed total iterations
        if self.benchmark_config['warmup_iterations'] >= self.benchmark_config['num_iterations']:
            raise constants.physics_parameter_error("warmup_iterations must be less than num_iterations")

    def set_network(self, network: neural_network, network_builder=None) -> None:
        """
        Set the network to be benchmarked.

        Args:
            network (neural_network): Neural network for benchmarking
            network_builder (optional): Associated network builder for parameter setting
        """
        self.current_network = network
        self.current_network_builder = network_builder

        # Also set network builder in accuracy analyzer for physics loss calculations
        if network_builder:
            self.accuracy_analyzer.set_network_builder(network_builder)

        logger.info("Set network for benchmarking")

    def benchmark_prediction_time(self, params: List[Union[int, float]],
                                 num_iterations: Optional[int] = None) -> Dict[str, float]:
        """
        Benchmark neural network prediction time for given parameters.

        The NN can handle any parameter values (coprime or non-coprime).
        Physics graph construction happens regardless to support benchmarking.

        Args:
            params (List[Union[int, float]]): TBG parameters for prediction
            num_iterations (int, optional): Number of timing iterations

        Returns:
            Dict[str, float]: Timing statistics in milliseconds

        Raises:
            constants.physics_parameter_error: If no network set or benchmarking fails
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network set. Call set_network() first.")
        if self.current_network_builder is None:
            raise constants.physics_parameter_error("Network builder required for parameter setting")

        num_iterations = num_iterations or self.benchmark_config['num_iterations']
        warmup_iterations = min(self.benchmark_config['warmup_iterations'], num_iterations // 3)

        try:
            # For non-coprime pairs, only set NN inputs (skip TBG graph construction)
            # Physics benchmarking will use the reduced coprime pair later
            a, b = int(params[0]), int(params[1])
            gcd_ab = np.gcd(a, b)

            if gcd_ab > 1:
                # Non-coprime: only set NN inputs without building TBG graphs
                self.current_network_builder.set_network_inputs_only(params)
                logger.debug(f"Non-coprime pair ({a},{b}): NN inputs only, skipping TBG graph construction")
            else:
                # Coprime: full parameter setting with TBG graph construction
                self.current_network_builder.set_network_parameters(params)

            prediction_times = []

            # Warmup runs (not counted)
            for _ in range(warmup_iterations):
                start_time = time.perf_counter()
                output = self.current_network.compute()
                time.perf_counter() - start_time

            # Actual timing runs
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                output = self.current_network.compute()
                prediction_time = time.perf_counter() - start_time
                prediction_times.append(prediction_time)
                self.nn_prediction_times.append(prediction_time)

            # Calculate statistics
            prediction_times_ms = [t * 1000 for t in prediction_times]

            timing_stats = {
                'avg_time_ms': np.mean(prediction_times_ms),
                'min_time_ms': np.min(prediction_times_ms),
                'max_time_ms': np.max(prediction_times_ms),
                'std_time_ms': np.std(prediction_times_ms),
                'median_time_ms': np.median(prediction_times_ms),
                'iterations': num_iterations,
                'prediction_result': output
            }

            logger.debug(f"NN prediction benchmark: {timing_stats['avg_time_ms']:.3f}ms avg")

            return timing_stats

        except Exception as e:
            logger.error(f"NN prediction benchmark failed: {str(e)}")
            raise constants.physics_parameter_error(f"Prediction benchmarking failed: {str(e)}")

    def benchmark_physics_computation(self, params: List[Union[int, float]], k_point: List[float],
                                    num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Benchmark physics-based Dirac point computation time using ADAM optimization.

        Runs ADAM optimizer from NN's k-point guess to find converged Dirac point.
        Also evaluates physics loss at the NN's initial guess point.

        Args:
            params (List[Union[int, float]]): TBG parameters
            k_point (List[float]): Initial k-point guess from NN
            num_iterations (int, optional): Number of timing iterations

        Returns:
            Dict[str, Any]: Timing statistics and computation results containing:
                - converged_k_x, converged_k_y: ADAM-optimized Dirac point coordinates
                - converged_velocity: Velocity at converged point
                - nn_point_loss: Physics loss at NN's initial guess
                - nn_point_gap, nn_point_r2, nn_point_isotropy: Metrics at NN point

        Raises:
            constants.physics_parameter_error: If benchmarking fails
        """
        if self.current_network_builder is None:
            raise constants.physics_parameter_error("Network builder required for physics computations")

        # Check if (a,b) are valid for physics computation
        a, b = int(params[0]), int(params[1])

        try:
            validate_ab(a, b)
            is_valid_for_physics = True
        except constants.physics_parameter_error as e:
            is_valid_for_physics = False
            logger.debug(f"Skipping physics benchmark for invalid pair ({a},{b}): {e}")
            return {
                'error': f'Invalid pair ({a},{b}) for physics - {str(e)}',
                'is_valid_for_physics': False
            }

        num_iterations = num_iterations or self.benchmark_config['num_iterations']
        warmup_iterations = min(self.benchmark_config['warmup_iterations'], num_iterations // 3)

        try:
            # Set parameters and initialize graphs (only for coprime pairs)
            self.current_network_builder.set_network_parameters(params)
            periodic_graph = self.current_network_builder.current_periodic_graph

            # Ensure k_point is a list of 2 floats
            if isinstance(k_point, tuple):
                k_point = list(k_point)
            k_point = [float(k) for k in k_point[:2]]

            # Get loss weights
            weights = constants.DEFAULT_NN_LOSS_WEIGHTS

            # First, evaluate physics loss at NN's prediction
            dirac_analyzer = Dirac_analysis(periodic_graph)
            nn_metrics, _, _, nn_velocity, _ = dirac_analyzer.check_Dirac_point(tuple(k_point), 1)
            nn_point_loss = np.sum(np.array(nn_metrics) * np.array(weights))

            computation_times = []
            converged_points = []

            # Warmup runs (ADAM now always returns best point found, no exception handling needed)
            for _ in range(warmup_iterations):
                start_time = time.perf_counter()
                converged_k, converged_vel = Find_Dirac_point(k_point, periodic_graph, weights)
                time.perf_counter() - start_time

            # Actual timing runs (ADAM logs warnings when it doesn't fully converge, but always returns best point)
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                converged_k, converged_vel = Find_Dirac_point(k_point, periodic_graph, weights)
                computation_time = time.perf_counter() - start_time

                computation_times.append(computation_time)
                self.physics_computation_times.append(computation_time)
                converged_points.append((converged_k, converged_vel))

            # Calculate statistics
            computation_times_ms = [t * 1000 for t in computation_times]

            # Use last converged result
            final_converged_k, final_converged_vel = converged_points[-1]

            timing_stats = {
                'avg_time_ms': np.mean(computation_times_ms),
                'min_time_ms': np.min(computation_times_ms),
                'max_time_ms': np.max(computation_times_ms),
                'std_time_ms': np.std(computation_times_ms),
                'median_time_ms': np.median(computation_times_ms),
                'iterations': num_iterations,
                # NEW FORMAT: k-coordinates and velocity from ADAM optimization
                'computation_results': [
                    final_converged_k[0],  # converged k_x
                    final_converged_k[1],  # converged k_y
                    final_converged_vel    # converged velocity
                ],
                # Also include NN point quality metrics
                'nn_point_loss': float(nn_point_loss),
                'nn_point_gap': float(nn_metrics[0]),
                'nn_point_r2': float(nn_metrics[1]),
                'nn_point_isotropy': float(nn_metrics[2])
            }

            logger.debug(f"Physics ADAM optimization: {timing_stats['avg_time_ms']:.3f}ms avg, "
                        f"k_distance={(np.linalg.norm([final_converged_k[0]-k_point[0], final_converged_k[1]-k_point[1]])):.4f}")

            return timing_stats

        except Exception as e:
            logger.error(f"Physics computation benchmark failed: {str(e)}")
            raise constants.physics_parameter_error(f"Physics benchmarking failed: {str(e)}")

    def calculate_acceleration_factor(self, test_params_list: List[List[Union[int, float]]],
                                    num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate acceleration factor comparing NN vs physics computation speeds.

        Args:
            test_params_list (List[List[Union[int, float]]]): List of parameter sets to test
            num_iterations (int, optional): Number of iterations per parameter set

        Returns:
            Dict[str, Any]: Comprehensive acceleration analysis results

        Raises:
            constants.physics_parameter_error: If calculation fails
        """
        if not test_params_list:
            raise constants.physics_parameter_error("No test parameters provided")

        num_iterations = num_iterations or self.benchmark_config['num_iterations']

        try:
            logger.info(f"Calculating acceleration factor with {len(test_params_list)} parameter sets")

            # Clear previous timing data
            self.nn_prediction_times.clear()
            self.physics_computation_times.clear()
            self.detailed_benchmark_results.clear()

            total_nn_time = 0.0
            total_physics_time = 0.0
            successful_comparisons = 0
            detailed_results = []

            for i, params in enumerate(test_params_list):
                logger.info(f"Testing parameter set {i+1}/{len(test_params_list)}: a={params[0]}, b={params[1]}")

                try:
                    # Benchmark NN prediction on original params
                    nn_stats = self.benchmark_prediction_time(params, num_iterations)
                    nn_output = nn_stats['prediction_result']

                    # Extract TWO predictions from network output [k_x1, k_y1, nu1, k_x2, k_y2, nu2]
                    if len(nn_output) >= 6:
                        k_point_guess_1 = tuple(nn_output[0:2])  # (k_x1, k_y1)
                        k_point_guess_2 = tuple(nn_output[3:5])  # (k_x2, k_y2)

                        # Calculate k-space separation between predictions
                        k_separation = np.sqrt((nn_output[0] - nn_output[3])**2 + (nn_output[1] - nn_output[4])**2)
                        logger.debug(f"  Two-point prediction: k-sep = {k_separation:.4f}")
                    else:
                        # Fallback for old single-prediction networks (backward compatibility)
                        k_point_guess_1 = tuple(nn_output[:2])
                        k_point_guess_2 = k_point_guess_1
                        k_separation = 0.0
                        logger.warning(f"  Network output has {len(nn_output)} elements (expected 6 for two-point). Using single prediction.")

                    # For non-coprime pairs, reduce to coprime equivalent for physics
                    a, b = int(params[0]), int(params[1])
                    gcd_ab = np.gcd(a, b)
                    physics_params = params  # Default: use original params
                    used_reduced_params = False

                    if gcd_ab > 1:
                        # Reduce to coprime equivalent: (a/gcd, b/gcd, same other params)
                        a_reduced, b_reduced = a // gcd_ab, b // gcd_ab
                        physics_params = [a_reduced, b_reduced] + list(params[2:])
                        used_reduced_params = True
                        logger.debug(f"  Non-coprime pair ({a},{b}) reduced to ({a_reduced},{b_reduced}) for physics")

                    # Benchmark physics computation TWICE - once from each NN prediction as initial guess
                    physics_stats_1 = self.benchmark_physics_computation(physics_params, k_point_guess_1, num_iterations)
                    physics_stats_2 = self.benchmark_physics_computation(physics_params, k_point_guess_2, num_iterations)

                    # Check if physics benchmarks succeeded
                    if 'error' in physics_stats_1 or 'error' in physics_stats_2:
                        # At least one physics run failed - skip
                        error_msg = physics_stats_1.get('error', '') or physics_stats_2.get('error', '')
                        logger.debug(f"  Skipping (invalid): {nn_stats['avg_time_ms']:.2f}ms, error: {error_msg}")
                        detailed_results.append({
                            'params': params,
                            'nn_time_ms': nn_stats['avg_time_ms'],
                            'physics_time_ms': None,
                            'speedup': None,
                            'nn_prediction': nn_stats['prediction_result'],
                            'physics_result_1': None,
                            'physics_result_2': None,
                            'k_separation': k_separation,
                            'note': error_msg
                        })
                        continue  # Skip to next parameter set

                    # Accumulate timing data - TWO physics runs per NN prediction
                    avg_nn_time = nn_stats['avg_time_ms'] / 1000  # Convert to seconds
                    avg_physics_time_1 = physics_stats_1['avg_time_ms'] / 1000
                    avg_physics_time_2 = physics_stats_2['avg_time_ms'] / 1000
                    total_physics_time_combined = avg_physics_time_1 + avg_physics_time_2

                    total_nn_time += avg_nn_time
                    total_physics_time += total_physics_time_combined  # Sum of both physics runs
                    successful_comparisons += 1

                    # Speedup: (2 physics runs) / (1 NN prediction)
                    speedup = total_physics_time_combined / avg_nn_time if avg_nn_time > 0 else 0

                    result_note = f"Scale-invariance test: NN on ({a},{b}), physics on ({physics_params[0]},{physics_params[1]})" if used_reduced_params else None

                    detailed_results.append({
                        'params': params,
                        'physics_params': physics_params if used_reduced_params else None,
                        'nn_time_ms': nn_stats['avg_time_ms'],
                        'physics_time_ms_1': physics_stats_1['avg_time_ms'],
                        'physics_time_ms_2': physics_stats_2['avg_time_ms'],
                        'physics_time_ms': total_physics_time_combined * 1000,  # Total for both runs
                        'speedup': speedup,
                        'nn_prediction': nn_stats['prediction_result'],
                        'physics_result_1': physics_stats_1['computation_results'],
                        'physics_result_2': physics_stats_2['computation_results'],
                        'k_separation': k_separation,
                        # NN point quality metrics for both predictions
                        'nn_point_1_loss': physics_stats_1.get('nn_point_loss'),
                        'nn_point_1_gap': physics_stats_1.get('nn_point_gap'),
                        'nn_point_1_r2': physics_stats_1.get('nn_point_r2'),
                        'nn_point_1_isotropy': physics_stats_1.get('nn_point_isotropy'),
                        'nn_point_2_loss': physics_stats_2.get('nn_point_loss'),
                        'nn_point_2_gap': physics_stats_2.get('nn_point_gap'),
                        'nn_point_2_r2': physics_stats_2.get('nn_point_r2'),
                        'nn_point_2_isotropy': physics_stats_2.get('nn_point_isotropy'),
                        'note': result_note
                    })

                    logger.info(f"  NN: {nn_stats['avg_time_ms']:.2f}ms, Physics: {physics_stats_1['avg_time_ms']:.2f}ms + {physics_stats_2['avg_time_ms']:.2f}ms = {total_physics_time_combined*1000:.2f}ms, Speedup: {speedup:.1f}x, k-sep: {k_separation:.4f}")

                except Exception as param_error:
                    logger.warning(f"Failed to benchmark parameter set {params}: {param_error}")
                    continue

            # Store detailed results for later access
            self.detailed_benchmark_results = detailed_results

            if successful_comparisons == 0:
                # All tests failed even after reduction - this is an error
                return {"error": "No successful benchmark comparisons"}

            # Calculate overall statistics
            avg_nn_time = total_nn_time / successful_comparisons
            avg_physics_time = total_physics_time / successful_comparisons
            acceleration_factor = avg_physics_time / avg_nn_time if avg_nn_time > 0 else 0

            # Log statistics using project utility
            self.stats.log_combination(
                duration=avg_nn_time,
                system_size=successful_comparisons,
                n_scale=acceleration_factor,
                success=True,
                no_intersection=True,
                num_of_Dirac=len(test_params_list)
            )

            # Perform comprehensive accuracy comparison using refactored module
            accuracy_analysis = self.accuracy_analyzer.compare_prediction_accuracy(detailed_results)

            results = {
                'acceleration_factor': acceleration_factor,
                'average_nn_time_ms': avg_nn_time * 1000,
                'average_physics_time_ms': avg_physics_time * 1000,
                'successful_comparisons': successful_comparisons,
                'total_parameter_sets': len(test_params_list),
                'iterations_per_set': num_iterations,
                'detailed_results': detailed_results,
                'statistics_summary': {
                    'nn_times_std': np.std([t * 1000 for t in self.nn_prediction_times]),
                    'physics_times_std': np.std([t * 1000 for t in self.physics_computation_times]),
                    'speedup_variance': np.var([r['speedup'] for r in detailed_results if r.get('speedup') is not None])
                },
                'accuracy_comparison': accuracy_analysis
            }

            logger.info(f"Acceleration factor calculation complete: {acceleration_factor:.1f}x speedup")
            if accuracy_analysis and 'k_point_error_avg' in accuracy_analysis:
                logger.info(f"  Average k-point error: {accuracy_analysis['k_point_error_avg']:.6f}")
                logger.info(f"  Average velocity error: {accuracy_analysis['velocity_error_avg']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Acceleration factor calculation failed: {str(e)}")
            raise constants.physics_parameter_error(f"Acceleration calculation failed: {str(e)}")

    def generate_performance_report(self, include_accuracy: bool = True,
                                   detailed_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis report.

        Args:
            include_accuracy (bool): Whether to include accuracy comparison (requires detailed_results)
            detailed_results (List[Dict], optional): Results from benchmark runs for accuracy analysis

        Returns:
            Dict[str, Any]: Detailed performance report
        """
        # Generate accuracy comparison if requested
        accuracy_comparison = None
        if include_accuracy and detailed_results:
            accuracy_comparison = self.accuracy_analyzer.compare_prediction_accuracy(detailed_results)

        # Use refactored report generator
        return self.report_generator.generate_performance_report(
            nn_prediction_times=self.nn_prediction_times,
            physics_computation_times=self.physics_computation_times,
            benchmark_config=self.benchmark_config,
            accuracy_comparison=accuracy_comparison
        )

    def save_benchmark_results(self, filename: Optional[str] = None) -> None:
        """
        Save comprehensive benchmark results including detailed statistics.

        Args:
            filename (str, optional): Output filename, uses default if None
        """
        if filename is None:
            filename = "benchmark_results"

        try:
            # Generate comprehensive performance report with accuracy comparison
            report = self.generate_performance_report(
                include_accuracy=True,
                detailed_results=self.detailed_benchmark_results
            )

            # Save CSV using project stats utility
            csv_filename = os.path.join(constants.PATH, f"{filename}.csv")
            self.stats.save_statistics(csv_filename)

            # Save detailed text log using refactored report generator
            log_filename = os.path.join(constants.PATH, f"{filename}_detailed_log.txt")
            self.report_generator.save_detailed_benchmark_log(
                filename=log_filename,
                report=report,
                nn_prediction_times=self.nn_prediction_times,
                physics_computation_times=self.physics_computation_times
            )

            logger.info(f"Benchmark results saved to {csv_filename}")
            logger.info(f"Detailed benchmark log saved to {log_filename}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
            raise constants.physics_parameter_error(f"Failed to save benchmark results: {str(e)}")

    def reset(self) -> None:
        """Reset benchmark state, clearing all timing data."""
        self.nn_prediction_times.clear()
        self.physics_computation_times.clear()
        self.detailed_benchmark_results.clear()
        self.stats.cleanup()
        self.current_network = None
        self.current_network_builder = None
        logger.info("dirac_network_benchmark state reset")


# Convenience functions following project patterns
def quick_benchmark(network, network_builder, test_params: List[Union[int, float]],
                   iterations: int = 5) -> Dict[str, float]:
    """
    Quick benchmark comparison for a single parameter set.

    Args:
        network: Neural network to benchmark
        network_builder: Associated network builder
        test_params: TBG parameters to test
        iterations: Number of benchmark iterations

    Returns:
        Dict[str, float]: Quick benchmark results

    Raises:
        constants.physics_parameter_error: If benchmarking fails
    """
    benchmark = dirac_network_benchmark({'num_iterations': iterations, 'warmup_iterations': 1})
    benchmark.set_network(network, network_builder)

    results = benchmark.calculate_acceleration_factor([test_params], iterations)

    if 'error' in results:
        return results

    return {
        'acceleration_factor': results['acceleration_factor'],
        'nn_time_ms': results['average_nn_time_ms'],
        'physics_time_ms': results['average_physics_time_ms']
    }
