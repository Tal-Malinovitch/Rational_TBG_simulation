"""
Performance benchmarking and acceleration analysis for Dirac point neural networks.

This module provides comprehensive benchmarking capabilities for measuring
neural network performance versus physics computations, following project
coding standards and integrating with existing utilities.

Classes:
    dirac_network_benchmark: Performance analysis and acceleration factor measurement
"""

import constants
from constants import np, time, logging
from constants import List, Dict, Tuple, Optional, Union, Any
from TBG import Dirac_analysis
from neural_network_base import neural_network
from stats import statistics

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
            # Set parameters
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
        
        Args:
            params (List[Union[int, float]]): TBG parameters
            k_point (List[float]): Initial k-point guess
            num_iterations (int, optional): Number of timing iterations
            
        Returns:
            Dict[str, Any]: Timing statistics and computation results
            
        Raises:
            constants.physics_parameter_error: If benchmarking fails
        """
        if self.current_network_builder is None:
            raise constants.physics_parameter_error("Network builder required for physics computations")
        
        num_iterations = num_iterations or self.benchmark_config['num_iterations']
        warmup_iterations = min(self.benchmark_config['warmup_iterations'], num_iterations // 3)
        
        try:
            # Set parameters and initialize graphs
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
                metrics, _, _, velocity_calc, _, _ = dirac_analyzer.check_Dirac_point(k_point, 1)
                time.perf_counter() - start_time
            
            # Actual timing runs
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                dirac_analyzer = Dirac_analysis(self.current_network_builder.current_periodic_graph)
                metrics, _, _, velocity_calc, _, _ = dirac_analyzer.check_Dirac_point(k_point, 1)
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
            
            total_nn_time = 0.0
            total_physics_time = 0.0
            successful_comparisons = 0
            detailed_results = []
            
            for i, params in enumerate(test_params_list):
                logger.info(f"Testing parameter set {i+1}/{len(test_params_list)}: a={params[0]}, b={params[1]}")
                
                try:
                    # Benchmark NN prediction
                    nn_stats = self.benchmark_prediction_time(params, num_iterations)
                    k_point_guess = tuple(nn_stats['prediction_result'][:2])  # Use NN prediction as physics guess
                    
                    # Benchmark physics computation
                    physics_stats = self.benchmark_physics_computation(params, k_point_guess, num_iterations)
                    
                    # Accumulate timing data
                    avg_nn_time = nn_stats['avg_time_ms'] / 1000  # Convert to seconds
                    avg_physics_time = physics_stats['avg_time_ms'] / 1000
                    
                    total_nn_time += avg_nn_time
                    total_physics_time += avg_physics_time
                    successful_comparisons += 1
                    
                    speedup = avg_physics_time / avg_nn_time if avg_nn_time > 0 else 0
                    
                    detailed_results.append({
                        'params': params,
                        'nn_time_ms': nn_stats['avg_time_ms'],
                        'physics_time_ms': physics_stats['avg_time_ms'],
                        'speedup': speedup,
                        'nn_prediction': nn_stats['prediction_result'],
                        'physics_result': physics_stats['computation_results']
                    })
                    
                    logger.info(f"  NN: {nn_stats['avg_time_ms']:.2f}ms, Physics: {physics_stats['avg_time_ms']:.2f}ms, Speedup: {speedup:.1f}x")
                    
                except Exception as param_error:
                    logger.warning(f"Failed to benchmark parameter set {params}: {param_error}")
                    continue
            
            if successful_comparisons == 0:
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
                    'speedup_variance': np.var([r['speedup'] for r in detailed_results])
                }
            }
            
            logger.info(f"Acceleration factor calculation complete: {acceleration_factor:.1f}x speedup")
            
            return results
            
        except Exception as e:
            logger.error(f"Acceleration factor calculation failed: {str(e)}")
            raise constants.physics_parameter_error(f"Acceleration calculation failed: {str(e)}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis report.
        
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
        
        return report
    
    def save_benchmark_results(self, filename: Optional[str] = None) -> None:
        """
        Save benchmark results using project statistics utility.
        
        Args:
            filename (str, optional): Output filename, uses default if None
        """
        if filename is None:
            filename = os.path.join(constants.PATH, "benchmark_results.csv")
        
        try:
            # Save using project stats utility
            self.stats.save_statistics(filename)
            logger.info(f"Benchmark results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
            raise constants.physics_parameter_error(f"Failed to save benchmark results: {str(e)}")
    
    def reset(self) -> None:
        """Reset benchmark state, clearing all timing data."""
        self.nn_prediction_times.clear()
        self.physics_computation_times.clear()
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