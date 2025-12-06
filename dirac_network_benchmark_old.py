"""
Performance benchmarking and acceleration analysis for Dirac point neural networks.

This module provides comprehensive benchmarking capabilities for measuring
neural network performance versus physics computations, following project
coding standards and integrating with existing utilities.

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
        Benchmark physics-based Dirac point computation time.

        Only works for coprime (a,b) pairs. Returns error dict for non-coprime pairs.

        Args:
            params (List[Union[int, float]]): TBG parameters
            k_point (List[float]): Initial k-point guess
            num_iterations (int, optional): Number of timing iterations

        Returns:
            Dict[str, Any]: Timing statistics and computation results,
                           or {'error': ...} for non-coprime pairs

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
            
            computation_times = []
            results = []
            
            # Ensure k_point is a tuple of 2 floats
            if not isinstance(k_point, tuple) or len(k_point) != 2:
                k_point = tuple(float(k) for k in k_point[:2])
            
            # Warmup runs
            for _ in range(warmup_iterations):
                start_time = time.perf_counter()
                dirac_analyzer = Dirac_analysis(self.current_network_builder.current_periodic_graph)
                metrics, _, _, velocity_calc, _ = dirac_analyzer.check_Dirac_point(k_point, 1)
                time.perf_counter() - start_time
            
            # Actual timing runs
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                dirac_analyzer = Dirac_analysis(self.current_network_builder.current_periodic_graph)
                metrics, _, _, velocity_calc, _ = dirac_analyzer.check_Dirac_point(k_point, 1)
                computation_time = time.perf_counter() - start_time
                
                computation_times.append(computation_time)
                self.physics_computation_times.append(computation_time)
                results.append(metrics + [velocity_calc])
            
            # Calculate statistics
            computation_times_ms = [t * 1000 for t in computation_times]
            
            timing_stats = {
                'avg_time_ms': np.mean(computation_times_ms),
                'min_time_ms': np.min(computation_times_ms),
                'max_time_ms': np.max(computation_times_ms),
                'std_time_ms': np.std(computation_times_ms),
                'median_time_ms': np.median(computation_times_ms),
                'iterations': num_iterations,
                'computation_results': results[-1]  # Last result
            }
            
            logger.debug(f"Physics computation benchmark: {timing_stats['avg_time_ms']:.3f}ms avg")
            
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
                    k_point_guess = tuple(nn_stats['prediction_result'][:2])  # Use NN prediction as physics guess

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

                    # Benchmark physics computation on (possibly reduced) params
                    physics_stats = self.benchmark_physics_computation(physics_params, k_point_guess, num_iterations)

                    # Check if physics benchmark succeeded
                    if 'error' in physics_stats:
                        # Still invalid even after reduction - skip
                        logger.debug(f"  Skipping (invalid): {nn_stats['avg_time_ms']:.2f}ms")
                        detailed_results.append({
                            'params': params,
                            'nn_time_ms': nn_stats['avg_time_ms'],
                            'physics_time_ms': None,
                            'speedup': None,
                            'nn_prediction': nn_stats['prediction_result'],
                            'physics_result': None,
                            'note': physics_stats['error']
                        })
                        continue  # Skip to next parameter set

                    # Accumulate timing data
                    avg_nn_time = nn_stats['avg_time_ms'] / 1000  # Convert to seconds
                    avg_physics_time = physics_stats['avg_time_ms'] / 1000

                    total_nn_time += avg_nn_time
                    total_physics_time += avg_physics_time
                    successful_comparisons += 1

                    speedup = avg_physics_time / avg_nn_time if avg_nn_time > 0 else 0

                    result_note = f"Scale-invariance test: NN on ({a},{b}), physics on ({physics_params[0]},{physics_params[1]})" if used_reduced_params else None

                    detailed_results.append({
                        'params': params,
                        'physics_params': physics_params if used_reduced_params else None,
                        'nn_time_ms': nn_stats['avg_time_ms'],
                        'physics_time_ms': physics_stats['avg_time_ms'],
                        'speedup': speedup,
                        'nn_prediction': nn_stats['prediction_result'],
                        'physics_result': physics_stats['computation_results'],
                        'note': result_note
                    })

                    logger.info(f"  NN: {nn_stats['avg_time_ms']:.2f}ms, Physics: {physics_stats['avg_time_ms']:.2f}ms, Speedup: {speedup:.1f}x")
                    
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

            # Perform comprehensive accuracy comparison between NN predictions and physics results
            accuracy_analysis = self._compare_prediction_accuracy(detailed_results)

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
        if not self.nn_prediction_times and not self.physics_computation_times:
            return {"error": "No benchmark data available. Run benchmarks first."}

        report = {
            'benchmark_summary': {
                'nn_predictions_count': len(self.nn_prediction_times),
                'physics_computations_count': len(self.physics_computation_times),
                'benchmark_config': self.benchmark_config.copy()
            }
        }

        # NN prediction statistics
        if self.nn_prediction_times:
            nn_times_ms = [t * 1000 for t in self.nn_prediction_times]
            report['nn_performance'] = {
                'avg_time_ms': np.mean(nn_times_ms),
                'min_time_ms': np.min(nn_times_ms),
                'max_time_ms': np.max(nn_times_ms),
                'std_time_ms': np.std(nn_times_ms),
                'median_time_ms': np.median(nn_times_ms),
                'percentile_95_ms': np.percentile(nn_times_ms, 95)
            }

        # Physics computation statistics
        if self.physics_computation_times:
            physics_times_ms = [t * 1000 for t in self.physics_computation_times]
            report['physics_performance'] = {
                'avg_time_ms': np.mean(physics_times_ms),
                'min_time_ms': np.min(physics_times_ms),
                'max_time_ms': np.max(physics_times_ms),
                'std_time_ms': np.std(physics_times_ms),
                'median_time_ms': np.median(physics_times_ms),
                'percentile_95_ms': np.percentile(physics_times_ms, 95)
            }

        # Overall acceleration analysis
        if self.nn_prediction_times and self.physics_computation_times:
            min_len = min(len(self.nn_prediction_times), len(self.physics_computation_times))
            acceleration_factors = [
                self.physics_computation_times[i] / self.nn_prediction_times[i]
                for i in range(min_len) if self.nn_prediction_times[i] > 0
            ]

            if acceleration_factors:
                report['acceleration_analysis'] = {
                    'mean_acceleration': np.mean(acceleration_factors),
                    'min_acceleration': np.min(acceleration_factors),
                    'max_acceleration': np.max(acceleration_factors),
                    'std_acceleration': np.std(acceleration_factors),
                    'median_acceleration': np.median(acceleration_factors),
                    'acceleration_factors': acceleration_factors
                }

        # Include accuracy comparison if requested and data available
        if include_accuracy and detailed_results:
            report['accuracy_comparison'] = self._compare_prediction_accuracy(detailed_results)

        return report
    
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

            # Save detailed text log
            log_filename = os.path.join(constants.PATH, f"{filename}_detailed_log.txt")
            self._save_detailed_benchmark_log(log_filename, report)

            logger.info(f"Benchmark results saved to {csv_filename}")
            logger.info(f"Detailed benchmark log saved to {log_filename}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
            raise constants.physics_parameter_error(f"Failed to save benchmark results: {str(e)}")

    def _save_detailed_benchmark_log(self, filename: str, report: Dict[str, Any]) -> None:
        """
        Save detailed benchmark statistics to a comprehensive text log file.

        Args:
            filename (str): Path to output log file
            report (dict): Performance report from generate_performance_report()
        """
        import os
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        with open(filename, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE BENCHMARK ANALYSIS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Check if we have error
            if "error" in report:
                f.write(f"ERROR: {report['error']}\n")
                return

            # Benchmark Configuration
            if 'benchmark_summary' in report:
                f.write("BENCHMARK CONFIGURATION\n")
                f.write("-" * 80 + "\n")
                summary = report['benchmark_summary']
                f.write(f"  NN Predictions Count:        {summary.get('nn_predictions_count', 0)}\n")
                f.write(f"  Physics Computations Count:  {summary.get('physics_computations_count', 0)}\n")
                if 'benchmark_config' in summary:
                    config = summary['benchmark_config']
                    f.write(f"  Iterations per Test:         {config.get('num_iterations', 'N/A')}\n")
                    f.write(f"  Warmup Iterations:           {config.get('warmup_iterations', 'N/A')}\n")
                    f.write(f"  Statistical Confidence:      {config.get('statistical_confidence', 'N/A')}\n")
                f.write("\n")

            # Neural Network Performance
            if 'nn_performance' in report:
                f.write("NEURAL NETWORK PERFORMANCE STATISTICS\n")
                f.write("-" * 80 + "\n")
                nn_perf = report['nn_performance']
                f.write(f"  Average Time:       {nn_perf.get('avg_time_ms', 0):.6f} ms\n")
                f.write(f"  Median Time:        {nn_perf.get('median_time_ms', 0):.6f} ms\n")
                f.write(f"  Minimum Time:       {nn_perf.get('min_time_ms', 0):.6f} ms\n")
                f.write(f"  Maximum Time:       {nn_perf.get('max_time_ms', 0):.6f} ms\n")
                f.write(f"  Standard Deviation: {nn_perf.get('std_time_ms', 0):.6f} ms\n")
                f.write(f"  95th Percentile:    {nn_perf.get('percentile_95_ms', 0):.6f} ms\n")
                f.write("\n")

            # Physics Computation Performance
            if 'physics_performance' in report:
                f.write("PHYSICS COMPUTATION PERFORMANCE STATISTICS\n")
                f.write("-" * 80 + "\n")
                phys_perf = report['physics_performance']
                f.write(f"  Average Time:       {phys_perf.get('avg_time_ms', 0):.6f} ms\n")
                f.write(f"  Median Time:        {phys_perf.get('median_time_ms', 0):.6f} ms\n")
                f.write(f"  Minimum Time:       {phys_perf.get('min_time_ms', 0):.6f} ms\n")
                f.write(f"  Maximum Time:       {phys_perf.get('max_time_ms', 0):.6f} ms\n")
                f.write(f"  Standard Deviation: {phys_perf.get('std_time_ms', 0):.6f} ms\n")
                f.write(f"  95th Percentile:    {phys_perf.get('percentile_95_ms', 0):.6f} ms\n")
                f.write("\n")

            # Acceleration Analysis
            if 'acceleration_analysis' in report:
                f.write("ACCELERATION ANALYSIS\n")
                f.write("-" * 80 + "\n")
                accel = report['acceleration_analysis']
                f.write(f"  Mean Acceleration Factor:    {accel.get('mean_acceleration', 0):.2f}x\n")
                f.write(f"  Median Acceleration Factor:  {accel.get('median_acceleration', 0):.2f}x\n")
                f.write(f"  Minimum Acceleration:        {accel.get('min_acceleration', 0):.2f}x\n")
                f.write(f"  Maximum Acceleration:        {accel.get('max_acceleration', 0):.2f}x\n")
                f.write(f"  Standard Deviation:          {accel.get('std_acceleration', 0):.4f}x\n")
                f.write("\n")

                # Individual acceleration factors
                if 'acceleration_factors' in accel and len(accel['acceleration_factors']) > 0:
                    f.write("  Individual Acceleration Factors:\n")
                    for i, factor in enumerate(accel['acceleration_factors'], 1):
                        f.write(f"    Test {i:2d}: {factor:.2f}x speedup\n")
                    f.write("\n")

            # Comparative Summary
            if 'nn_performance' in report and 'physics_performance' in report:
                f.write("COMPARATIVE SUMMARY\n")
                f.write("-" * 80 + "\n")
                nn_avg = report['nn_performance'].get('avg_time_ms', 0)
                phys_avg = report['physics_performance'].get('avg_time_ms', 0)
                if nn_avg > 0:
                    overall_speedup = phys_avg / nn_avg
                    time_saved_per_prediction = phys_avg - nn_avg
                    f.write(f"  Overall Speedup:              {overall_speedup:.2f}x\n")
                    f.write(f"  Time Saved per Prediction:    {time_saved_per_prediction:.3f} ms\n")
                    f.write(f"  Efficiency Gain:              {((overall_speedup - 1) / overall_speedup * 100):.2f}%\n")

                    # Extrapolation
                    predictions_per_hour_nn = 3600000 / nn_avg if nn_avg > 0 else 0
                    predictions_per_hour_phys = 3600000 / phys_avg if phys_avg > 0 else 0
                    f.write(f"\n  Predictions per Hour:\n")
                    f.write(f"    Neural Network:             {predictions_per_hour_nn:,.0f}\n")
                    f.write(f"    Physics Computation:        {predictions_per_hour_phys:,.0f}\n")
                    f.write(f"    Additional Throughput:      {predictions_per_hour_nn - predictions_per_hour_phys:,.0f}\n")
                f.write("\n")

            # Accuracy Comparison - Most Important Section
            if 'accuracy_comparison' in report and 'error' not in report.get('accuracy_comparison', {}):
                f.write("PREDICTION ACCURACY ANALYSIS\n")
                f.write("=" * 80 + "\n")
                accuracy = report['accuracy_comparison']

                f.write(f"  Number of Comparisons: {accuracy.get('num_comparisons', 0)}\n\n")

                # K_x Error Statistics
                if 'k_x_error' in accuracy:
                    f.write("  k_x Coordinate Error:\n")
                    k_x = accuracy['k_x_error']
                    f.write(f"    Mean Error:         {k_x.get('mean', 0):.8f}\n")
                    f.write(f"    Median Error:       {k_x.get('median', 0):.8f}\n")
                    f.write(f"    Std Deviation:      {k_x.get('std', 0):.8f}\n")
                    f.write(f"    Min Error:          {k_x.get('min', 0):.8f}\n")
                    f.write(f"    Max Error:          {k_x.get('max', 0):.8f}\n")
                    f.write(f"    95th Percentile:    {k_x.get('percentile_95', 0):.8f}\n\n")

                # K_y Error Statistics
                if 'k_y_error' in accuracy:
                    f.write("  k_y Coordinate Error:\n")
                    k_y = accuracy['k_y_error']
                    f.write(f"    Mean Error:         {k_y.get('mean', 0):.8f}\n")
                    f.write(f"    Median Error:       {k_y.get('median', 0):.8f}\n")
                    f.write(f"    Std Deviation:      {k_y.get('std', 0):.8f}\n")
                    f.write(f"    Min Error:          {k_y.get('min', 0):.8f}\n")
                    f.write(f"    Max Error:          {k_y.get('max', 0):.8f}\n")
                    f.write(f"    95th Percentile:    {k_y.get('percentile_95', 0):.8f}\n\n")

                # K Magnitude Error Statistics
                if 'k_magnitude_error' in accuracy:
                    f.write("  k-Point Magnitude Error (Euclidean Distance):\n")
                    k_mag = accuracy['k_magnitude_error']
                    f.write(f"    Mean Error:         {k_mag.get('mean', 0):.8f}\n")
                    f.write(f"    Median Error:       {k_mag.get('median', 0):.8f}\n")
                    f.write(f"    Std Deviation:      {k_mag.get('std', 0):.8f}\n")
                    f.write(f"    Min Error:          {k_mag.get('min', 0):.8f}\n")
                    f.write(f"    Max Error:          {k_mag.get('max', 0):.8f}\n")
                    f.write(f"    95th Percentile:    {k_mag.get('percentile_95', 0):.8f}\n\n")

                # Velocity Error Statistics
                if 'velocity_error' in accuracy:
                    f.write("  Velocity (v) Absolute Error:\n")
                    vel = accuracy['velocity_error']
                    f.write(f"    Mean Error:         {vel.get('mean', 0):.6f}\n")
                    f.write(f"    Median Error:       {vel.get('median', 0):.6f}\n")
                    f.write(f"    Std Deviation:      {vel.get('std', 0):.6f}\n")
                    f.write(f"    Min Error:          {vel.get('min', 0):.6f}\n")
                    f.write(f"    Max Error:          {vel.get('max', 0):.6f}\n")
                    f.write(f"    95th Percentile:    {vel.get('percentile_95', 0):.6f}\n\n")

                # Velocity Relative Error Statistics
                if 'velocity_relative_error_percent' in accuracy:
                    f.write("  Velocity (v) Relative Error (%):\n")
                    vel_rel = accuracy['velocity_relative_error_percent']
                    f.write(f"    Mean Error:         {vel_rel.get('mean', 0):.4f}%\n")
                    f.write(f"    Median Error:       {vel_rel.get('median', 0):.4f}%\n")
                    f.write(f"    Std Deviation:      {vel_rel.get('std', 0):.4f}%\n")
                    f.write(f"    Min Error:          {vel_rel.get('min', 0):.4f}%\n")
                    f.write(f"    Max Error:          {vel_rel.get('max', 0):.4f}%\n")
                    f.write(f"    95th Percentile:    {vel_rel.get('percentile_95', 0):.4f}%\n\n")

                # Physics Loss Function Comparison
                if 'physics_loss_at_nn_prediction' in accuracy:
                    f.write("  Physics Loss at NN Predicted Point:\n")
                    nn_loss = accuracy['physics_loss_at_nn_prediction']
                    f.write(f"    Mean:               {nn_loss.get('mean', 0):.8f}\n")
                    f.write(f"    Median:             {nn_loss.get('median', 0):.8f}\n")
                    f.write(f"    Std Deviation:      {nn_loss.get('std', 0):.8f}\n")
                    f.write(f"    Min:                {nn_loss.get('min', 0):.8f}\n")
                    f.write(f"    Max:                {nn_loss.get('max', 0):.8f}\n")
                    f.write(f"    95th Percentile:    {nn_loss.get('percentile_95', 0):.8f}\n\n")

                if 'physics_loss_at_physics_result' in accuracy:
                    f.write("  Physics Loss at Physics Result Point:\n")
                    phys_loss = accuracy['physics_loss_at_physics_result']
                    f.write(f"    Mean:               {phys_loss.get('mean', 0):.8f}\n")
                    f.write(f"    Median:             {phys_loss.get('median', 0):.8f}\n")
                    f.write(f"    Std Deviation:      {phys_loss.get('std', 0):.8f}\n")
                    f.write(f"    Min:                {phys_loss.get('min', 0):.8f}\n")
                    f.write(f"    Max:                {phys_loss.get('max', 0):.8f}\n")
                    f.write(f"    95th Percentile:    {phys_loss.get('percentile_95', 0):.8f}\n\n")

                if 'physics_loss_ratio' in accuracy:
                    f.write("  Physics Loss Ratio (NN/Physics):\n")
                    f.write(f"    {accuracy['physics_loss_ratio'].get('description', '')}\n")
                    loss_ratio = accuracy['physics_loss_ratio']
                    f.write(f"    Mean:               {loss_ratio.get('mean', 0):.6f}\n")
                    f.write(f"    Median:             {loss_ratio.get('median', 0):.6f}\n")
                    f.write(f"    Std Deviation:      {loss_ratio.get('std', 0):.6f}\n")
                    f.write(f"    Min:                {loss_ratio.get('min', 0):.6f}\n")
                    f.write(f"    Max:                {loss_ratio.get('max', 0):.6f}\n")
                    f.write(f"    95th Percentile:    {loss_ratio.get('percentile_95', 0):.6f}\n\n")

                # Detailed Per-Test Comparison
                if 'comparison_details' in accuracy:
                    f.write("\nDETAILED PER-TEST COMPARISON\n")
                    f.write("-" * 80 + "\n")
                    for i, detail in enumerate(accuracy['comparison_details'], 1):
                        f.write(f"\n  Test {i}: Parameters {detail.get('params', [])}\n")
                        f.write(f"    NN Prediction:      k=({detail.get('nn_k_x', 0):.6f}, {detail.get('nn_k_y', 0):.6f}), v={detail.get('nn_velocity', 0):.4f}\n")
                        f.write(f"    Physics Result:     k=({detail.get('physics_k_x', 0):.6f}, {detail.get('physics_k_y', 0):.6f}), v={detail.get('physics_velocity', 0):.4f}\n")
                        f.write(f"    k_x Error:          {detail.get('k_x_error', 0):.8f}\n")
                        f.write(f"    k_y Error:          {detail.get('k_y_error', 0):.8f}\n")
                        f.write(f"    k Magnitude Error:  {detail.get('k_magnitude_error', 0):.8f}\n")
                        f.write(f"    Velocity Error:     {detail.get('velocity_error', 0):.6f}\n")
                        f.write(f"    Velocity Rel Error: {detail.get('velocity_relative_error_percent', 0):.4f}%\n")
                        if detail.get('nn_physics_loss') is not None:
                            f.write(f"    Physics Loss @ NN:  {detail.get('nn_physics_loss', 0):.8f}\n")
                        if detail.get('physics_physics_loss') is not None:
                            f.write(f"    Physics Loss @ Opt: {detail.get('physics_physics_loss', 0):.8f}\n")
                        if detail.get('loss_ratio') is not None:
                            f.write(f"    Loss Ratio:         {detail.get('loss_ratio', 0):.6f}\n")

                f.write("\n")

            # Raw Data Summary
            f.write("RAW TIMING DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total NN Predictions:        {len(self.nn_prediction_times)}\n")
            f.write(f"  Total Physics Computations:  {len(self.physics_computation_times)}\n")
            if self.nn_prediction_times:
                f.write(f"\n  All NN Prediction Times (ms):\n")
                for i, t in enumerate(self.nn_prediction_times, 1):
                    f.write(f"    {i:4d}: {t*1000:.6f}\n")
            if self.physics_computation_times:
                f.write(f"\n  All Physics Computation Times (ms):\n")
                for i, t in enumerate(self.physics_computation_times, 1):
                    f.write(f"    {i:4d}: {t*1000:.6f}\n")

            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF BENCHMARK ANALYSIS\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Detailed benchmark log written to {filename}")
    
    def _compare_prediction_accuracy(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare accuracy of neural network predictions vs physics computations.

        Analyzes differences in k-point coordinates (k_x, k_y) and velocity predictions,
        providing comprehensive error metrics and statistical analysis.

        Args:
            detailed_results (List[Dict]): List of benchmark results with nn_prediction and physics_result

        Returns:
            Dict[str, Any]: Comprehensive accuracy comparison metrics
        """
        if not detailed_results:
            return {'error': 'No results to compare'}

        # Extract predictions and physics results
        k_x_errors = []
        k_y_errors = []
        k_magnitude_errors = []
        velocity_errors = []
        velocity_relative_errors = []
        log_velocity_errors = []
        nu_errors = []
        nu_relative_errors = []

        comparison_details = []

        for result in detailed_results:
            try:
                # NN prediction: [k_x, k_y, nu] (where nu = 1/(1+v))
                nn_pred = result.get('nn_prediction', [])
                # Physics result: [k_x, k_y, ..., velocity_calc]
                phys_result = result.get('physics_result', [])

                if len(nn_pred) >= 3 and len(phys_result) >= 4:
                    # Extract k-points
                    nn_k_x = nn_pred[0]
                    nn_k_y = nn_pred[1]
                    nn_nu = nn_pred[2]

                    phys_k_x = phys_result[0]
                    phys_k_y = phys_result[1]
                    phys_velocity = phys_result[3]  # velocity is 4th element

                    # Convert NN's nu back to velocity: v = (1-nu)/nu
                    nn_nu_clamped = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, nn_nu))
                    nn_velocity = (1.0 - nn_nu_clamped) / nn_nu_clamped

                    # Calculate errors
                    k_x_error = abs(nn_k_x - phys_k_x)
                    k_y_error = abs(nn_k_y - phys_k_y)
                    k_magnitude_error = np.sqrt((nn_k_x - phys_k_x)**2 + (nn_k_y - phys_k_y)**2)
                    velocity_error = abs(nn_velocity - phys_velocity)

                    # Velocity error metrics (multiple for different perspectives)
                    # Training data: median=0.18, 95th=1.34, velocities can vanish at Dirac points

                    # 1. Standard relative error (kept for reference, but problematic for small v)
                    velocity_relative_error = velocity_error / abs(phys_velocity) if abs(phys_velocity) > 1e-10 else 0

                    # 2. LOG-SCALE ERROR: Natural for quantities spanning orders of magnitude
                    #    Treats multiplicative errors fairly, handles v→0 gracefully
                    epsilon = 1e-6
                    log_velocity_error = abs(np.log(abs(nn_velocity) + epsilon) - np.log(abs(phys_velocity) + epsilon))

                    # 3. NU-SPACE ERROR: Error in the NN's native output space
                    #    nu = 1/(1+v), NN directly predicts nu, bounded in (0,1)
                    phys_nu = 1.0 / (1.0 + abs(phys_velocity))
                    nu_error = abs(nn_nu - phys_nu)
                    nu_relative_error = nu_error / abs(phys_nu) if abs(phys_nu) > 1e-10 else 0

                    k_x_errors.append(k_x_error)
                    k_y_errors.append(k_y_error)
                    k_magnitude_errors.append(k_magnitude_error)
                    velocity_errors.append(velocity_error)
                    velocity_relative_errors.append(velocity_relative_error)
                    log_velocity_errors.append(log_velocity_error)
                    nu_errors.append(nu_error)
                    nu_relative_errors.append(nu_relative_error)

                    # Calculate physics loss at both points if we have network builder
                    nn_physics_loss = None
                    physics_physics_loss = None
                    loss_ratio = None

                    if self.current_network_builder and self.current_network_builder.current_periodic_graph:
                        try:
                            # Calculate physics loss at NN predicted point
                            dirac_analyzer = Dirac_analysis(self.current_network_builder.current_periodic_graph)
                            nn_metrics, _, _, nn_vel_calc, _ = dirac_analyzer.check_Dirac_point((nn_k_x, nn_k_y), 1)

                            # Use default loss weights
                            loss_weights = constants.DEFAULT_NN_LOSS_WEIGHTS
                            nn_dirac_loss = np.sum(np.array(nn_metrics) * np.array(loss_weights))
                            nn_velocity_loss = nn_vel_calc - nn_velocity
                            nn_physics_loss = nn_dirac_loss**2 + nn_velocity_loss**2

                            # Calculate physics loss at physics computation point
                            phys_metrics, _, _, phys_vel_calc, _ = dirac_analyzer.check_Dirac_point((phys_k_x, phys_k_y), 1)
                            phys_dirac_loss = np.sum(np.array(phys_metrics) * np.array(loss_weights))
                            phys_velocity_loss = phys_vel_calc - phys_velocity
                            physics_physics_loss = phys_dirac_loss**2 + phys_velocity_loss**2

                            # Calculate ratio and absolute difference
                            # IMPORTANT: Ratio can be misleading when both losses are small
                            # e.g., physics_loss=1e-10, nn_loss=1e-8 gives ratio=100 but both are excellent
                            loss_absolute_diff = nn_physics_loss - physics_physics_loss

                            # Only compute ratio if physics loss is significant (> 1e-6)
                            # Otherwise, ratio is not meaningful
                            if physics_physics_loss > 1e-6:
                                loss_ratio = nn_physics_loss / physics_physics_loss
                            else:
                                # Both losses very small - ratio not meaningful
                                loss_ratio = None

                        except Exception as e:
                            logger.warning(f"Failed to compute physics loss comparison: {e}")

                    comparison_details.append({
                        'params': result.get('params', []),
                        # Raw values (always saved for post-analysis)
                        'nn_k_x': nn_k_x,
                        'nn_k_y': nn_k_y,
                        'nn_nu': nn_nu,  # Raw NN output
                        'nn_velocity': nn_velocity,
                        'physics_k_x': phys_k_x,
                        'physics_k_y': phys_k_y,
                        'physics_velocity': phys_velocity,
                        'physics_nu': phys_nu,  # Converted from physics velocity
                        # K-point errors
                        'k_x_error': k_x_error,
                        'k_y_error': k_y_error,
                        'k_magnitude_error': k_magnitude_error,
                        # Velocity errors (multiple metrics)
                        'velocity_error': velocity_error,  # Absolute error
                        'velocity_relative_error_percent': velocity_relative_error * 100,  # Standard relative
                        'log_velocity_error': log_velocity_error,  # Log-scale error
                        'nu_error': nu_error,  # Nu-space absolute error
                        'nu_relative_error_percent': nu_relative_error * 100,  # Nu-space relative error
                        # Physics loss
                        'nn_physics_loss': nn_physics_loss,
                        'physics_physics_loss': physics_physics_loss,
                        'loss_ratio': loss_ratio,
                        'loss_absolute_diff': loss_absolute_diff if 'loss_absolute_diff' in locals() else None
                    })

            except Exception as e:
                logger.warning(f"Failed to compare result: {e}")
                continue

        if not k_x_errors:
            return {'error': 'No valid comparisons could be made'}

        # Collect physics loss data
        nn_physics_losses = [d['nn_physics_loss'] for d in comparison_details if d.get('nn_physics_loss') is not None]
        physics_physics_losses = [d['physics_physics_loss'] for d in comparison_details if d.get('physics_physics_loss') is not None]
        loss_ratios = [d['loss_ratio'] for d in comparison_details if d.get('loss_ratio') is not None]
        loss_absolute_diffs = [d['loss_absolute_diff'] for d in comparison_details if d.get('loss_absolute_diff') is not None]

        # Compute comprehensive statistics
        accuracy_metrics = {
            'num_comparisons': len(k_x_errors),
            'k_x_error': {
                'mean': np.mean(k_x_errors),
                'median': np.median(k_x_errors),
                'std': np.std(k_x_errors),
                'min': np.min(k_x_errors),
                'max': np.max(k_x_errors),
                'percentile_95': np.percentile(k_x_errors, 95)
            },
            'k_y_error': {
                'mean': np.mean(k_y_errors),
                'median': np.median(k_y_errors),
                'std': np.std(k_y_errors),
                'min': np.min(k_y_errors),
                'max': np.max(k_y_errors),
                'percentile_95': np.percentile(k_y_errors, 95)
            },
            'k_magnitude_error': {
                'mean': np.mean(k_magnitude_errors),
                'median': np.median(k_magnitude_errors),
                'std': np.std(k_magnitude_errors),
                'min': np.min(k_magnitude_errors),
                'max': np.max(k_magnitude_errors),
                'percentile_95': np.percentile(k_magnitude_errors, 95)
            },
            'velocity_error': {
                'mean': np.mean(velocity_errors),
                'median': np.median(velocity_errors),
                'std': np.std(velocity_errors),
                'min': np.min(velocity_errors),
                'max': np.max(velocity_errors),
                'percentile_95': np.percentile(velocity_errors, 95)
            },
            'velocity_relative_error_percent': {
                'mean': np.mean(velocity_relative_errors) * 100,
                'median': np.median(velocity_relative_errors) * 100,
                'std': np.std(velocity_relative_errors) * 100,
                'min': np.min(velocity_relative_errors) * 100,
                'max': np.max(velocity_relative_errors) * 100,
                'percentile_95': np.percentile(velocity_relative_errors, 95) * 100
            },
            'log_velocity_error': {
                'mean': np.mean(log_velocity_errors),
                'median': np.median(log_velocity_errors),
                'std': np.std(log_velocity_errors),
                'min': np.min(log_velocity_errors),
                'max': np.max(log_velocity_errors),
                'percentile_95': np.percentile(log_velocity_errors, 95),
                'description': 'Log-scale error: abs(log(nn_v + eps) - log(phys_v + eps)), natural for quantities spanning orders of magnitude'
            },
            'nu_error': {
                'mean': np.mean(nu_errors),
                'median': np.median(nu_errors),
                'std': np.std(nu_errors),
                'min': np.min(nu_errors),
                'max': np.max(nu_errors),
                'percentile_95': np.percentile(nu_errors, 95),
                'description': 'Nu-space absolute error where nu=1/(1+v), error in NN native output space'
            },
            'nu_relative_error_percent': {
                'mean': np.mean(nu_relative_errors) * 100,
                'median': np.median(nu_relative_errors) * 100,
                'std': np.std(nu_relative_errors) * 100,
                'min': np.min(nu_relative_errors) * 100,
                'max': np.max(nu_relative_errors) * 100,
                'percentile_95': np.percentile(nu_relative_errors, 95) * 100
            },
            'comparison_details': comparison_details,
            # Legacy keys for backward compatibility
            'k_point_error_avg': np.mean(k_magnitude_errors),
            'velocity_error_avg': np.mean(velocity_errors)
        }

        # Add physics loss statistics if available
        if nn_physics_losses:
            accuracy_metrics['physics_loss_at_nn_prediction'] = {
                'mean': np.mean(nn_physics_losses),
                'median': np.median(nn_physics_losses),
                'std': np.std(nn_physics_losses),
                'min': np.min(nn_physics_losses),
                'max': np.max(nn_physics_losses),
                'percentile_95': np.percentile(nn_physics_losses, 95)
            }

        if physics_physics_losses:
            accuracy_metrics['physics_loss_at_physics_result'] = {
                'mean': np.mean(physics_physics_losses),
                'median': np.median(physics_physics_losses),
                'std': np.std(physics_physics_losses),
                'min': np.min(physics_physics_losses),
                'max': np.max(physics_physics_losses),
                'percentile_95': np.percentile(physics_physics_losses, 95)
            }

        if loss_ratios:
            accuracy_metrics['physics_loss_ratio'] = {
                'mean': np.mean(loss_ratios),
                'median': np.median(loss_ratios),
                'std': np.std(loss_ratios),
                'min': np.min(loss_ratios),
                'max': np.max(loss_ratios),
                'percentile_95': np.percentile(loss_ratios, 95),
                'num_ratios_computed': len(loss_ratios),
                'total_tests': len(comparison_details),
                'description': 'Ratio of physics loss at NN prediction vs physics result. Only computed when physics loss > 1e-6 to avoid misleading ratios from tiny losses.'
            }

        if loss_absolute_diffs:
            accuracy_metrics['physics_loss_absolute_diff'] = {
                'mean': np.mean(loss_absolute_diffs),
                'median': np.median(loss_absolute_diffs),
                'std': np.std(loss_absolute_diffs),
                'min': np.min(loss_absolute_diffs),
                'max': np.max(loss_absolute_diffs),
                'percentile_95': np.percentile(loss_absolute_diffs, 95),
                'description': 'Absolute difference: nn_physics_loss - physics_physics_loss (positive means NN worse, closer to 0 is better)'
            }

        # Binned error analysis - group by system size and velocity magnitude
        binned_analysis = self._compute_binned_error_analysis(comparison_details)
        if binned_analysis:
            accuracy_metrics['binned_analysis'] = binned_analysis

        return accuracy_metrics

    def _compute_binned_error_analysis(self, comparison_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute binned error analysis grouped by system characteristics.

        Args:
            comparison_details: List of comparison detail dicts

        Returns:
            Dict with binned statistics
        """
        if not comparison_details:
            return {}

        # Note: We can't bin by N here because it's not in comparison_details
        # We could add it, but for now let's bin by velocity magnitude

        binned_stats = {}

        # Bin by velocity magnitude (small, medium, large)
        velocity_bins = {
            'small': {'range': (0, 1), 'k_errors': [], 'v_errors': []},
            'medium': {'range': (1, 10), 'k_errors': [], 'v_errors': []},
            'large': {'range': (10, float('inf')), 'k_errors': [], 'v_errors': []}
        }

        for detail in comparison_details:
            phys_v = abs(detail.get('physics_velocity', 0))
            k_err = detail.get('k_magnitude_error')
            v_err = detail.get('velocity_error')

            if k_err is None or v_err is None:
                continue

            # Assign to bin
            for bin_name, bin_data in velocity_bins.items():
                v_min, v_max = bin_data['range']
                if v_min <= phys_v < v_max:
                    bin_data['k_errors'].append(k_err)
                    bin_data['v_errors'].append(v_err)
                    break

        # Compute statistics for each bin
        for bin_name, bin_data in velocity_bins.items():
            if bin_data['k_errors']:
                binned_stats[f'velocity_{bin_name}'] = {
                    'velocity_range': bin_data['range'],
                    'count': len(bin_data['k_errors']),
                    'k_error_mean': np.mean(bin_data['k_errors']),
                    'k_error_median': np.median(bin_data['k_errors']),
                    'v_error_mean': np.mean(bin_data['v_errors']),
                    'v_error_median': np.median(bin_data['v_errors'])
                }

        return binned_stats

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