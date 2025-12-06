"""
Comprehensive Benchmark Executor

Handles execution of comprehensive benchmark suites for trained Dirac point neural networks.
This module manages model loading, parallel test execution, and results collection.

Classes:
    comprehensive_benchmark_executor: Executes benchmark tests and collects results

Functions:
    _run_single_test_worker: Worker function for parallel test execution
"""

import constants
from constants import np, logging, os, time, json
from constants import Dict, List, Any, Tuple
from dirac_network_benchmark import dirac_network_benchmark
from dirac_network_persistence import dirac_network_persistence
from dirac_network_builder import dirac_network_builder
import multiprocessing as mp

# Configure logging
logger = logging.getLogger(__name__)


def _run_single_test_worker(args: Tuple[int, List[float], str, int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Worker function to run a single benchmark test in parallel.

    This function is called by multiprocessing.Pool to execute tests in parallel.
    Each worker creates its own network instance and runs the benchmark independently.

    Args:
        args: Tuple of (test_index, params, model_path, num_iterations, training_target_or_none)

    Returns:
        Dict containing test results with success status, timing, and accuracy metrics
    """
    test_index, params, model_path, num_iterations, training_target = args

    try:
        # Create fresh instances for this worker process
        network_builder = dirac_network_builder()
        nn = network_builder.build_network()

        persistence = dirac_network_persistence()
        persistence.set_network(nn)
        result = persistence.load_network_weights(model_path)

        if 'error' in result:
            return {
                'test_index': test_index,
                'params': params,
                'success': False,
                'error': f'Failed to load network: {result.get("error")}'
            }

        benchmark = dirac_network_benchmark()
        benchmark.set_network(nn, network_builder)

        # Compute system size metrics using REDUCED coprime parameters
        # The benchmark reduces non-coprime pairs before building TBG system
        from utils import compute_twist_constants
        a, b = int(params[0]), int(params[1])
        gcd_ab = int(np.gcd(a, b))
        a_reduced, b_reduced = int(a // gcd_ab), int(b // gcd_ab)
        N_value, alpha, factor, k_point = compute_twist_constants(a_reduced, b_reduced)

        # Run benchmark
        benchmark_result = benchmark.calculate_acceleration_factor(
            test_params_list=[params],
            num_iterations=num_iterations
        )

        # Extract key metrics
        if 'error' not in benchmark_result:
            result_entry = {
                'test_index': test_index,
                'params': params,
                'system_size': {
                    'N': float(N_value),
                    'num_nodes': len(network_builder.current_periodic_graph.nodes) if network_builder.current_periodic_graph else None
                },
                'acceleration_factor': benchmark_result['acceleration_factor'],
                'nn_time_ms': benchmark_result['average_nn_time_ms'],
                'physics_time_ms': benchmark_result['average_physics_time_ms'],
                'success': True
            }

            # Add accuracy metrics if available
            if 'accuracy_comparison' in benchmark_result and 'error' not in benchmark_result['accuracy_comparison']:
                accuracy = benchmark_result['accuracy_comparison']
                result_entry['accuracy'] = {
                    'k_magnitude_error_mean': accuracy.get('k_magnitude_error', {}).get('mean', None),
                    'velocity_error_mean': accuracy.get('velocity_error', {}).get('mean', None),
                    'velocity_rel_error_percent': accuracy.get('velocity_relative_error_percent', {}).get('mean', None),
                    'num_comparisons': accuracy.get('num_comparisons', 0)
                }

                # Add k-space separation for two-point predictions
                if 'k_separation' in accuracy:
                    result_entry['accuracy']['k_separation_mean'] = accuracy['k_separation'].get('mean', None)

                # Add detailed predictions (NN and physics results) from comparison_details
                if 'comparison_details' in accuracy and len(accuracy['comparison_details']) > 0:
                    result_entry['detailed_predictions'] = []
                    for detail in accuracy['comparison_details']:
                        # Convert numpy arrays/scalars to Python types for JSON serialization
                        nn_nu_val = detail['nn_nu']
                        if isinstance(nn_nu_val, np.ndarray):
                            nn_nu_val = float(nn_nu_val.item()) if nn_nu_val.size == 1 else float(nn_nu_val[0])
                        else:
                            nn_nu_val = float(nn_nu_val)

                        pred_detail = {
                            'prediction_index': int(detail['prediction_index']),
                            'nn_k': [float(detail['nn_k_x']), float(detail['nn_k_y'])],
                            'nn_nu': nn_nu_val,
                            'nn_velocity': float(detail['nn_velocity']),
                            'physics_k': [float(detail['physics_k_x']), float(detail['physics_k_y'])],
                            'physics_velocity': float(detail['physics_velocity']),
                            'k_error': float(detail['k_magnitude_error']),
                            'v_error': float(detail['velocity_error'])
                        }
                        result_entry['detailed_predictions'].append(pred_detail)

            # If this is a training sample, add expected target
            if training_target is not None:
                result_entry['training_target'] = training_target

            return result_entry
        else:
            return {
                'test_index': test_index,
                'params': params,
                'success': False,
                'error': benchmark_result.get('error', 'Unknown error')
            }

    except Exception as e:
        return {
            'test_index': test_index,
            'params': params,
            'success': False,
            'error': str(e)
        }


class comprehensive_benchmark_executor:
    """
    Executes comprehensive benchmarks on trained neural networks with parallel execution.

    This class handles loading models and benchmark suites, then executes
    tests in parallel across multiple categories (interpolation, extrapolation,
    redundancy pairs, edge cases).

    Attributes:
        model_path (str): Path to trained model weights
        benchmark_suite_path (str): Path to benchmark test suite JSON
        benchmark_suite (dict): Loaded benchmark suite data
        nn (neural_network): Loaded neural network
        benchmark (dirac_network_benchmark): Benchmark module instance
        results (dict): Results storage for all test categories
    """

    def __init__(self, model_path: str = "final_trained_model.npz",
                 benchmark_suite_path: str = None):
        """
        Initialize benchmark executor.

        Args:
            model_path: Path to trained model weights
            benchmark_suite_path: Path to benchmark test suite JSON
        """
        self.model_path = model_path

        if benchmark_suite_path is None:
            benchmark_suite_path = os.path.join(constants.PATH, "benchmark_test_suite.json")

        self.benchmark_suite_path = benchmark_suite_path
        self.benchmark_suite = None
        self.nn = None
        self.network_builder = None
        self.benchmark = None

        # Results storage
        self.results = {
            'training_samples': [],
            'weight_interpolation': [],
            'weight_extrapolation': [],
            'threshold_extrapolation': [],
            'scaling_invariance': [],
            'coprime_extrapolation_close': [],
            'coprime_extrapolation_far': [],
            'redundancy_pairs': []
        }

        # Checkpoint settings
        self.checkpoint_file = os.path.join(constants.PATH, "comprehensive_benchmark_checkpoint.json")
        self.checkpoint_interval = constants.BENCHMARK_CHECKPOINT_INTERVAL

        # Track benchmark configuration for checkpoint validation
        self.num_iterations = None
        self.num_processes = None

        logger.info(f"comprehensive_benchmark_executor initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Suite: {benchmark_suite_path}")

    def load_model(self) -> bool:
        """
        Load the trained neural network model.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading trained model...")

        model_full_path = os.path.join(constants.PATH, self.model_path)
        if not os.path.exists(model_full_path):
            logger.error(f"Model not found: {model_full_path}")
            return False

        # Build neural network architecture
        self.network_builder = dirac_network_builder()
        self.nn = self.network_builder.build_network()

        # Load weights from file
        persistence = dirac_network_persistence()
        persistence.set_network(self.nn)
        result = persistence.load_network_weights(self.model_path)
        success = 'error' not in result

        if success:
            # Create benchmark module with loaded network and builder
            self.benchmark = dirac_network_benchmark()
            self.benchmark.set_network(self.nn, network_builder=self.network_builder)
            logger.info("Model loaded successfully!")
        else:
            logger.error("Failed to load model")

        return success

    def load_benchmark_suite(self) -> bool:
        """
        Load the benchmark test suite from JSON file.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading benchmark suite...")

        if not os.path.exists(self.benchmark_suite_path):
            logger.error(f"Benchmark suite not found: {self.benchmark_suite_path}")
            return False

        with open(self.benchmark_suite_path, 'r') as f:
            self.benchmark_suite = json.load(f)

        # Count total tests
        total_tests = (len(self.benchmark_suite.get('weight_interpolation', [])) +
                      len(self.benchmark_suite.get('weight_extrapolation', [])) +
                      len(self.benchmark_suite.get('threshold_extrapolation', [])) +
                      len(self.benchmark_suite.get('scaling_invariance', [])) +
                      len(self.benchmark_suite.get('coprime_extrapolation_close', [])) +
                      len(self.benchmark_suite.get('coprime_extrapolation_far', [])) +
                      sum(len(pair) for pair in self.benchmark_suite['redundancy_pairs']))

        logger.info(f"Benchmark suite loaded: {total_tests} parameter sets")
        logger.info(f"  - Weight interpolation: {len(self.benchmark_suite.get('weight_interpolation', []))}")
        logger.info(f"  - Weight extrapolation: {len(self.benchmark_suite.get('weight_extrapolation', []))}")
        logger.info(f"  - Threshold extrapolation: {len(self.benchmark_suite.get('threshold_extrapolation', []))}")
        logger.info(f"  - Scaling invariance: {len(self.benchmark_suite.get('scaling_invariance', []))}")
        logger.info(f"  - Coprime extrap (close): {len(self.benchmark_suite.get('coprime_extrapolation_close', []))}")
        logger.info(f"  - Coprime extrap (far): {len(self.benchmark_suite.get('coprime_extrapolation_far', []))}")
        logger.info(f"  - Redundancy pairs: {len(self.benchmark_suite['redundancy_pairs'])}")

        return True

    def save_checkpoint(self) -> None:
        """
        Save current benchmark progress to checkpoint file.

        Saves all test results, configuration, and metadata to allow resumption
        after interruption or crash.
        """
        checkpoint_data = {
            'model_path': self.model_path,
            'benchmark_suite_path': self.benchmark_suite_path,
            'num_iterations': self.num_iterations,
            'num_processes': self.num_processes,
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved: {len(self.results['training_samples'])} training, "
                       f"{len(self.results['weight_interpolation'])} weight_interp, "
                       f"{len(self.results['weight_extrapolation'])} weight_extrap, "
                       f"{len(self.results['threshold_extrapolation'])} thresh_extrap, "
                       f"{len(self.results['scaling_invariance'])} scaling, "
                       f"{len(self.results['coprime_extrapolation_close'])} coprime_close, "
                       f"{len(self.results['coprime_extrapolation_far'])} coprime_far, "
                       f"{len(self.results['redundancy_pairs'])} redundancy")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if it exists and validate configuration compatibility.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise

        Raises:
            Warning logged if checkpoint configuration differs from current run
        """
        if not os.path.exists(self.checkpoint_file):
            return False

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Verify checkpoint is for the same model and suite
            if (checkpoint_data['model_path'] != self.model_path or
                checkpoint_data['benchmark_suite_path'] != self.benchmark_suite_path):
                logger.warning("Checkpoint is for different model/suite, ignoring")
                return False

            # Restore results
            self.results = checkpoint_data['results']

            # Restore configuration (for validation in run_full_benchmark)
            self.num_iterations = checkpoint_data.get('num_iterations')
            self.num_processes = checkpoint_data.get('num_processes')

            logger.info("=" * 80)
            logger.info("RESUMING FROM CHECKPOINT")
            logger.info("=" * 80)
            logger.info(f"Checkpoint from: {checkpoint_data['timestamp']}")
            logger.info(f"Completed tests:")
            logger.info(f"  - Training samples: {len(self.results.get('training_samples', []))}")
            logger.info(f"  - Weight interpolation: {len(self.results.get('weight_interpolation', []))}")
            logger.info(f"  - Weight extrapolation: {len(self.results.get('weight_extrapolation', []))}")
            logger.info(f"  - Threshold extrapolation: {len(self.results.get('threshold_extrapolation', []))}")
            logger.info(f"  - Scaling invariance: {len(self.results.get('scaling_invariance', []))}")
            logger.info(f"  - Coprime extrap (close): {len(self.results.get('coprime_extrapolation_close', []))}")
            logger.info(f"  - Coprime extrap (far): {len(self.results.get('coprime_extrapolation_far', []))}")
            logger.info(f"  - Redundancy pairs: {len(self.results.get('redundancy_pairs', []))}")
            logger.info("=" * 80)

            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def run_full_benchmark(self, num_iterations: int = constants.BENCHMARK_NUM_ITERATIONS, num_processes: int = None) -> Dict[str, Any]:
        """
        Run the complete benchmark suite in parallel.

        Args:
            num_iterations: Number of timing iterations per parameter set
            num_processes: Number of parallel processes (default: CPU count - 1)

        Returns:
            Dict with comprehensive benchmark results
        """
        if self.nn is None:
            if not self.load_model():
                return {'error': 'Failed to load model'}

        if self.benchmark_suite is None:
            if not self.load_benchmark_suite():
                return {'error': 'Failed to load benchmark suite'}

        # Try to load checkpoint
        checkpoint_loaded = self.load_checkpoint()

        # Store configuration for this run
        if not checkpoint_loaded:
            self.num_iterations = num_iterations
            # Determine number of parallel processes with proper limits
            if num_processes is None:
                # Use DEFAULT_NUM_PROCESSES logic: min(cpu_count - RESERVED_CORES, MAX_PARALLEL_PROCESSES)
                if constants.DEFAULT_NUM_PROCESSES is not None:
                    num_processes = constants.DEFAULT_NUM_PROCESSES
                else:
                    num_processes = min(
                        max(1, mp.cpu_count() - constants.RESERVED_CORES),
                        constants.MAX_PARALLEL_PROCESSES
                    )
            else:
                # Enforce maximum limit even if explicitly provided
                num_processes = min(num_processes, constants.MAX_PARALLEL_PROCESSES)

            self.num_processes = num_processes
            logger.info(f"Using {num_processes} parallel processes (MAX_PARALLEL_PROCESSES={constants.MAX_PARALLEL_PROCESSES}, "
                       f"CPU_COUNT={mp.cpu_count()}, RESERVED_CORES={constants.RESERVED_CORES})")
        else:
            # Validate configuration matches checkpoint
            if self.num_iterations != num_iterations:
                logger.warning(f"num_iterations mismatch: checkpoint={self.num_iterations}, requested={num_iterations}. Using checkpoint value.")
                num_iterations = self.num_iterations

            # Use checkpoint's num_processes if none specified, but enforce limits
            if num_processes is None:
                if self.num_processes is not None:
                    # Use checkpoint value but enforce maximum limit
                    num_processes = min(self.num_processes, constants.MAX_PARALLEL_PROCESSES)
                    if self.num_processes > constants.MAX_PARALLEL_PROCESSES:
                        logger.warning(f"Checkpoint num_processes={self.num_processes} exceeds MAX_PARALLEL_PROCESSES={constants.MAX_PARALLEL_PROCESSES}. "
                                     f"Reducing to {num_processes} for system stability.")
                else:
                    # Calculate default with limits
                    if constants.DEFAULT_NUM_PROCESSES is not None:
                        num_processes = constants.DEFAULT_NUM_PROCESSES
                    else:
                        num_processes = min(
                            max(1, mp.cpu_count() - constants.RESERVED_CORES),
                            constants.MAX_PARALLEL_PROCESSES
                        )
            else:
                # Enforce maximum limit even if explicitly provided
                num_processes = min(num_processes, constants.MAX_PARALLEL_PROCESSES)
                if self.num_processes is not None and num_processes != min(self.num_processes, constants.MAX_PARALLEL_PROCESSES):
                    logger.warning(f"num_processes mismatch: checkpoint={self.num_processes}, requested={num_processes}. Using requested value.")

            self.num_processes = num_processes
            logger.info(f"Resuming with {num_processes} parallel processes (MAX_PARALLEL_PROCESSES={constants.MAX_PARALLEL_PROCESSES})")

        logger.info("=" * 80)
        if not checkpoint_loaded:
            logger.info("STARTING COMPREHENSIVE BENCHMARK")
        else:
            logger.info("CONTINUING COMPREHENSIVE BENCHMARK")
        logger.info("=" * 80)

        start_time = time.time()

        logger.info(f"Configuration: {num_iterations} iterations, {num_processes} parallel processes")

        # Run each category in parallel (skip if already completed)
        if len(self.results['training_samples']) < len(self.benchmark_suite['training_samples']):
            logger.info("\n0. Running training sample tests (baseline)...")
            self._run_test_category('training_samples', self.benchmark_suite['training_samples'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n0. Training samples: ALREADY COMPLETED ({len(self.results['training_samples'])} tests)")

        if len(self.results['weight_interpolation']) < len(self.benchmark_suite.get('weight_interpolation', [])):
            logger.info("\n1. Running weight interpolation tests...")
            self._run_test_category('weight_interpolation', self.benchmark_suite['weight_interpolation'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n1. Weight interpolation: ALREADY COMPLETED ({len(self.results['weight_interpolation'])} tests)")

        if len(self.results['weight_extrapolation']) < len(self.benchmark_suite.get('weight_extrapolation', [])):
            logger.info("\n2. Running weight extrapolation tests...")
            self._run_test_category('weight_extrapolation', self.benchmark_suite['weight_extrapolation'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n2. Weight extrapolation: ALREADY COMPLETED ({len(self.results['weight_extrapolation'])} tests)")

        if len(self.results['threshold_extrapolation']) < len(self.benchmark_suite.get('threshold_extrapolation', [])):
            logger.info("\n3. Running threshold extrapolation tests...")
            self._run_test_category('threshold_extrapolation', self.benchmark_suite['threshold_extrapolation'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n3. Threshold extrapolation: ALREADY COMPLETED ({len(self.results['threshold_extrapolation'])} tests)")

        if len(self.results['scaling_invariance']) < len(self.benchmark_suite.get('scaling_invariance', [])):
            logger.info("\n4. Running scaling invariance tests...")
            self._run_test_category('scaling_invariance', self.benchmark_suite['scaling_invariance'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n4. Scaling invariance: ALREADY COMPLETED ({len(self.results['scaling_invariance'])} tests)")

        if len(self.results['coprime_extrapolation_close']) < len(self.benchmark_suite.get('coprime_extrapolation_close', [])):
            logger.info("\n5. Running close coprime extrapolation tests...")
            self._run_test_category('coprime_extrapolation_close', self.benchmark_suite['coprime_extrapolation_close'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n5. Close coprime extrapolation: ALREADY COMPLETED ({len(self.results['coprime_extrapolation_close'])} tests)")

        if len(self.results['coprime_extrapolation_far']) < len(self.benchmark_suite.get('coprime_extrapolation_far', [])):
            logger.info("\n6. Running far coprime extrapolation tests...")
            self._run_test_category('coprime_extrapolation_far', self.benchmark_suite['coprime_extrapolation_far'],
                                   num_iterations, num_processes)
        else:
            logger.info(f"\n6. Far coprime extrapolation: ALREADY COMPLETED ({len(self.results['coprime_extrapolation_far'])} tests)")

        if len(self.results['redundancy_pairs']) < len(self.benchmark_suite['redundancy_pairs']):
            logger.info("\n7. Running redundancy pair tests...")
            self._run_redundancy_tests(self.benchmark_suite['redundancy_pairs'], num_iterations)
        else:
            logger.info(f"\n7. Redundancy pairs: ALREADY COMPLETED ({len(self.results['redundancy_pairs'])} tests)")

        total_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Total runtime: {total_time/60:.1f} minutes")
        logger.info("=" * 80)

        # Compile results
        final_results = {
            'metadata': {
                'model_path': self.model_path,
                'benchmark_suite_path': self.benchmark_suite_path,
                'num_iterations': num_iterations,
                'num_processes': num_processes,
                'total_runtime_seconds': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'training_samples': self.results['training_samples'],
            'weight_interpolation': self.results['weight_interpolation'],
            'weight_extrapolation': self.results['weight_extrapolation'],
            'threshold_extrapolation': self.results['threshold_extrapolation'],
            'scaling_invariance': self.results['scaling_invariance'],
            'coprime_extrapolation_close': self.results['coprime_extrapolation_close'],
            'coprime_extrapolation_far': self.results['coprime_extrapolation_far'],
            'redundancy_pairs': self.results['redundancy_pairs']
        }

        return final_results

    def _run_test_category(self, category: str, test_params_list: List[List],
                          num_iterations: int, num_processes: int = None) -> None:
        """
        Run a category of tests in parallel with checkpoint support.

        Args:
            category: Category name ('training_samples', 'interpolation', 'extrapolation', or 'edge_cases')
            test_params_list: List of parameter sets to test
            num_iterations: Number of timing iterations
            num_processes: Number of parallel processes (default: CPU count - 1)
        """
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        # Calculate how many tests remain (in case resuming from checkpoint)
        already_completed = len(self.results[category])
        total_tests = len(test_params_list)

        if already_completed > 0:
            logger.info(f"Resuming {category}: {already_completed}/{total_tests} already completed")
            # Skip already completed tests
            test_params_list = test_params_list[already_completed:]

        if len(test_params_list) == 0:
            logger.info(f"All {category} tests already completed")
            return

        logger.info(f"Running {len(test_params_list)} {category} tests in parallel using {num_processes} processes...")

        # Get training targets if this is the training_samples category
        training_targets = {}
        if category == 'training_samples' and 'training_sample_targets' in self.benchmark_suite:
            training_targets = self.benchmark_suite['training_sample_targets']

        # Prepare worker arguments (adjust indices for remaining tests)
        worker_args = [
            (i + already_completed, params, self.model_path, num_iterations, training_targets.get(str(i + already_completed), None))
            for i, params in enumerate(test_params_list, 1)
        ]

        # Run tests in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Use imap_unordered for better performance (results as they complete)
            results_iterator = pool.imap_unordered(_run_single_test_worker, worker_args)

            # Process results as they arrive
            completed = 0
            last_checkpoint = 0  # Track last checkpoint to avoid missing saves with imap_unordered
            for result in results_iterator:
                self.results[category].append(result)
                completed += 1

                # Progress logging and checkpoint saving
                total_completed = already_completed + completed
                if completed % 10 == 0 or total_completed == total_tests:
                    logger.info(f"  Progress: {total_completed}/{total_tests} tests completed")

                # Save checkpoint periodically (use >= to handle out-of-order results from imap_unordered)
                if completed - last_checkpoint >= self.checkpoint_interval:
                    self.save_checkpoint()
                    last_checkpoint = completed

        # Final checkpoint save for this category
        self.save_checkpoint()

        # Sort results by test_index to maintain order
        self.results[category].sort(key=lambda x: x.get('test_index', 0))

        logger.info(f"Completed {category}: {len(self.results[category])}/{total_tests} tests")

    def _run_redundancy_tests(self, redundancy_pairs: List[List[List]],
                             num_iterations: int) -> None:
        """
        Run redundancy pair tests (scale-equivalent systems) with checkpoint support.

        These pairs should give IDENTICAL physics results since they represent
        the same physical system with the same graph size.

        Args:
            redundancy_pairs: List of parameter pairs to test
            num_iterations: Number of timing iterations
        """
        # Calculate how many tests remain (in case resuming from checkpoint)
        already_completed = len(self.results['redundancy_pairs'])
        total_tests = len(redundancy_pairs)

        if already_completed > 0:
            logger.info(f"Resuming redundancy pairs: {already_completed}/{total_tests} already completed")

        logger.info(f"Running {total_tests - already_completed} redundancy pair tests...")

        for i, pair in enumerate(redundancy_pairs, 1):
            # Skip already completed tests
            if i <= already_completed:
                continue

            try:
                params_1, params_2 = pair

                # Test both systems
                result_1 = self.benchmark.calculate_acceleration_factor(
                    test_params_list=[params_1],
                    num_iterations=num_iterations
                )

                result_2 = self.benchmark.calculate_acceleration_factor(
                    test_params_list=[params_2],
                    num_iterations=num_iterations
                )

                # Store pair result
                pair_result = {
                    'pair_index': i,
                    'params_1': params_1,
                    'params_2': params_2,
                    'result_1': result_1,
                    'result_2': result_2,
                    'success': 'error' not in result_1 and 'error' not in result_2
                }

                self.results['redundancy_pairs'].append(pair_result)

                # Progress logging and checkpoint saving
                if i % 5 == 0 or i == total_tests:
                    logger.info(f"  Progress: {i}/{total_tests} pairs completed")

                # Save checkpoint periodically
                if (i - already_completed) % self.checkpoint_interval == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Redundancy pair {i} failed with exception: {e}")
                self.results['redundancy_pairs'].append({
                    'pair_index': i,
                    'params_1': params_1 if 'params_1' in locals() else None,
                    'params_2': params_2 if 'params_2' in locals() else None,
                    'success': False,
                    'error': str(e)
                })

        # Final checkpoint save for redundancy tests
        self.save_checkpoint()

        logger.info(f"Completed redundancy pairs: {len(self.results['redundancy_pairs'])}/{total_tests} tests")
