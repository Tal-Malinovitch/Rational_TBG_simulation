"""
Comprehensive Benchmark Design for Dirac Point Neural Network

This script analyzes the training data parameter space and generates a rigorous
benchmark test suite that:
1. Samples diverse parameter combinations
2. Includes physical redundancies (scale-equivalent systems like a=7,b=1 vs a=14,b=2)
   NOTE: These redundant parameters give the SAME graph size - that's the physical symmetry!
3. Tests interpolation (within training range) and extrapolation (outside training range)
4. Provides statistical significance with sufficient sample size

Training data analysis shows:
- 211,518 total samples
- 486 unique (a,b) pairs
- ~15 scale-redundant groups (e.g., ratio 1.167: (7,6), (14,12), (21,18), etc.)
- 12 unique weight combinations
- 16 unique threshold combinations

The benchmark suite will be used to:
- Baseline performance measurement (data-trained model)
- Comparison after physics-based training
- Quantify improvements in accuracy and generalization
"""

import constants
from constants import np, logging, os
from constants import List, Dict, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)


class benchmark_test_suite:
    """
    Comprehensive benchmark test suite designer for TBG Dirac point predictions.

    This class analyzes training data to create a statistically rigorous benchmark
    that tests model performance across diverse parameter combinations, including
    physical redundancies and edge cases.
    """

    def __init__(self, training_data_path: str = None):
        """
        Initialize benchmark suite designer.

        Args:
            training_data_path: Path to training data CSV file
        """
        if training_data_path is None:
            training_data_path = os.path.join(constants.PATH, "Training_data", "dirac_training_data.csv")

        self.training_data_path = training_data_path
        self.training_data = []
        self.parameter_stats = {}

        logger.info(f"Initialized benchmark_test_suite with data: {training_data_path}")

    def analyze_training_data(self) -> Dict[str, Any]:
        """
        Analyze training data to understand parameter space coverage.

        Returns:
            Dict containing parameter statistics and distributions
        """
        import csv
        from collections import defaultdict

        logger.info("Loading and analyzing training data...")

        # Load training data
        with open(self.training_data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.training_data.append({
                    'a': int(row['a']),
                    'b': int(row['b']),
                    'interlayer_dist_threshold': float(row['interlayer_dist_threshold']),
                    'intralayer_dist_threshold': float(row['intralayer_dist_threshold']),
                    'inter_graph_weight': float(row['inter_graph_weight']),
                    'intra_graph_weight': float(row['intra_graph_weight']),
                    'N_scale': float(row['N_scale']),
                    'num_nodes': int(row['num_nodes']),
                    'target_k_x': float(row['target_k_x']),
                    'target_k_y': float(row['target_k_y']),
                    'Dirac_velocity': float(row['Dirac_velocity']),
                })

        logger.info(f"Loaded {len(self.training_data):,} training samples")

        # Analyze parameter ranges
        param_ranges = {}
        for param in ['a', 'b', 'interlayer_dist_threshold', 'intralayer_dist_threshold',
                     'inter_graph_weight', 'intra_graph_weight']:
            values = [row[param] for row in self.training_data]
            unique_values = list(set(values))
            param_ranges[param] = {
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(np.mean(values)),
                'unique_values': len(unique_values),
                'unique_list': sorted(unique_values)
            }

        # Analyze (a,b) combinations
        ab_pairs = defaultdict(int)
        for row in self.training_data:
            ab_pairs[(row['a'], row['b'])] += 1

        # Find scale-equivalent systems (same a/b ratio = same physics, same graph size)
        ratio_groups = defaultdict(list)
        for ab in ab_pairs.keys():
            ratio = round(ab[0] / ab[1], 6)  # Round to avoid floating point issues
            ratio_groups[ratio].append(ab)

        # Keep only groups with multiple representations
        scale_equivalent_groups = {ratio: pairs for ratio, pairs in ratio_groups.items() if len(pairs) > 1}

        # Weight and threshold combinations
        weight_combos = set()
        threshold_combos = set()
        for row in self.training_data:
            weight_combos.add((round(row['inter_graph_weight'], 4), round(row['intra_graph_weight'], 4)))
            threshold_combos.add((round(row['interlayer_dist_threshold'], 4), round(row['intralayer_dist_threshold'], 4)))

        self.parameter_stats = {
            'total_samples': len(self.training_data),
            'parameter_ranges': param_ranges,
            'unique_ab_pairs': len(ab_pairs),
            'ab_pairs_dict': dict(ab_pairs),
            'scale_equivalent_groups': {k: v for k, v in scale_equivalent_groups.items()},
            'num_scale_groups': len(scale_equivalent_groups),
            'weight_combinations': sorted(list(weight_combos)),
            'threshold_combinations': sorted(list(threshold_combos)),
            'num_weight_combos': len(weight_combos),
            'num_threshold_combos': len(threshold_combos)
        }

        logger.info(f"Found {self.parameter_stats['num_scale_groups']} scale-equivalent groups")
        logger.info(f"Found {self.parameter_stats['num_weight_combos']} weight combinations")
        logger.info(f"Found {self.parameter_stats['num_threshold_combos']} threshold combinations")

        return self.parameter_stats

    def generate_benchmark_suite(self,
                                 num_redundancy_tests: int = None) -> Dict[str, List[List]]:
        """
        Generate comprehensive benchmark test suite.

        Args:
            num_redundancy_tests: Number of scale-equivalent system pairs to test (default from constants)

        Returns:
            Dict with different test categories and their parameter sets
        """
        # Use constants if not specified
        if num_redundancy_tests is None:
            num_redundancy_tests = constants.BENCHMARK_NUM_REDUNDANCY_TESTS

        if not self.training_data:
            self.analyze_training_data()

        logger.info("Generating comprehensive benchmark suite...")

        benchmark_suite = {
            'training_samples': [],
            'weight_interpolation': [],
            'weight_extrapolation': [],
            'threshold_extrapolation': [],
            'scaling_invariance': [],
            'coprime_extrapolation_close': [],
            'coprime_extrapolation_far': [],
            'redundancy_pairs': []
        }

        # 0. TRAINING SAMPLES - Exact samples from training data (baseline performance)
        num_training_samples = 50
        logger.info(f"Generating {num_training_samples} training sample tests...")
        training_sample_tests, training_targets = self._generate_training_sample_tests(num_training_samples)
        benchmark_suite['training_samples'] = training_sample_tests
        benchmark_suite['training_sample_targets'] = training_targets

        # 1. WEIGHT INTERPOLATION - Weights between training values
        num_weight_interp = 50
        logger.info(f"Generating {num_weight_interp} weight interpolation tests...")
        benchmark_suite['weight_interpolation'] = self._generate_weight_interpolation_tests(num_weight_interp)

        # 2. WEIGHT EXTRAPOLATION - Weights above training max
        num_weight_extrap = 50
        logger.info(f"Generating {num_weight_extrap} weight extrapolation tests...")
        benchmark_suite['weight_extrapolation'] = self._generate_weight_extrapolation_tests(num_weight_extrap)

        # 3. THRESHOLD EXTRAPOLATION - Thresholds above training max
        num_thresh_extrap = 50
        logger.info(f"Generating {num_thresh_extrap} threshold extrapolation tests...")
        benchmark_suite['threshold_extrapolation'] = self._generate_threshold_extrapolation_tests(num_thresh_extrap)

        # 4. SCALING INVARIANCE - Large scaling factors [10,20]
        num_scaling = 30
        logger.info(f"Generating {num_scaling} scaling invariance tests...")
        benchmark_suite['scaling_invariance'] = self._generate_scaling_invariance_tests(num_scaling)

        # 5. COPRIME EXTRAPOLATION CLOSE - New coprime a∈[11,15]
        num_coprime_close = 40
        logger.info(f"Generating {num_coprime_close} close coprime extrapolation tests...")
        benchmark_suite['coprime_extrapolation_close'] = self._generate_coprime_extrapolation_close_tests(num_coprime_close)

        # 6. COPRIME EXTRAPOLATION FAR - New coprime a∈[16,20]
        num_coprime_far = 20
        logger.info(f"Generating {num_coprime_far} far coprime extrapolation tests...")
        benchmark_suite['coprime_extrapolation_far'] = self._generate_coprime_extrapolation_far_tests(num_coprime_far)

        # 7. REDUNDANCY TESTS - Physical equivalences (scale-equivalent systems)
        logger.info(f"Generating {num_redundancy_tests} redundancy pair tests...")
        redundancy_pairs = self._generate_redundancy_tests(num_redundancy_tests)
        benchmark_suite['redundancy_pairs'] = redundancy_pairs

        # Calculate total tests
        total_individual_tests = (len(benchmark_suite['training_samples']) +
                                 len(benchmark_suite['weight_interpolation']) +
                                 len(benchmark_suite['weight_extrapolation']) +
                                 len(benchmark_suite['threshold_extrapolation']) +
                                 len(benchmark_suite['scaling_invariance']) +
                                 len(benchmark_suite['coprime_extrapolation_close']) +
                                 len(benchmark_suite['coprime_extrapolation_far']) +
                                 sum(len(pair) for pair in benchmark_suite['redundancy_pairs']))

        logger.info(f"Benchmark suite generated: {total_individual_tests} total parameter sets")
        logger.info(f"  - Training samples: {len(benchmark_suite['training_samples'])}")
        logger.info(f"  - Weight interpolation: {len(benchmark_suite['weight_interpolation'])}")
        logger.info(f"  - Weight extrapolation: {len(benchmark_suite['weight_extrapolation'])}")
        logger.info(f"  - Threshold extrapolation: {len(benchmark_suite['threshold_extrapolation'])}")
        logger.info(f"  - Scaling invariance: {len(benchmark_suite['scaling_invariance'])}")
        logger.info(f"  - Coprime extrap (close): {len(benchmark_suite['coprime_extrapolation_close'])}")
        logger.info(f"  - Coprime extrap (far): {len(benchmark_suite['coprime_extrapolation_far'])}")
        logger.info(f"  - Redundancy pairs: {len(benchmark_suite['redundancy_pairs'])} pairs ({sum(len(pair) for pair in benchmark_suite['redundancy_pairs'])} tests)")

        return benchmark_suite

    def _generate_training_sample_tests(self, num_tests: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Generate test cases by sampling EXACT parameters from training data.

        This provides a baseline: if the network can't predict training data well,
        it indicates a training problem. If training data is perfect but
        interpolation/extrapolation fail, it indicates generalization issues.

        Args:
            num_tests: Number of training samples to select

        Returns:
            Tuple of (test parameter sets, training targets metadata)
        """
        tests = []
        training_targets = {}  # Maps test_index -> {target_k_x, target_k_y, target_velocity, N_scale, num_nodes}

        np.random.seed(41)  # Different seed for training samples

        # Sample random indices from training data
        total_samples = len(self.training_data)
        if num_tests > total_samples:
            logger.warning(f"Requested {num_tests} samples but only {total_samples} available")
            num_tests = total_samples

        # Sample without replacement to get diverse cases
        sampled_indices = np.random.choice(total_samples, size=num_tests, replace=False)

        for test_idx, data_idx in enumerate(sampled_indices, 1):
            sample = self.training_data[data_idx]
            # Extract parameters: [a, b, interlayer_thresh, intralayer_thresh, inter_weight, intra_weight]
            # Training data format: a, b, interlayer_dist_threshold, intralayer_dist_threshold,
            #                       inter_graph_weight, intra_graph_weight, N_scale, num_nodes,
            #                       target_k_x, target_k_y, Dirac_velocity
            test_params = [
                int(sample['a']),      # a
                int(sample['b']),      # b
                float(sample['interlayer_dist_threshold']),    # interlayer_dist_threshold
                float(sample['intralayer_dist_threshold']),    # intralayer_dist_threshold
                float(sample['inter_graph_weight']),    # inter_graph_weight
                float(sample['intra_graph_weight'])     # intra_graph_weight
            ]
            tests.append(test_params)

            # Store training targets for this test
            training_targets[str(test_idx)] = {
                'target_k_x': float(sample['target_k_x']),
                'target_k_y': float(sample['target_k_y']),
                'target_velocity': float(sample['Dirac_velocity']),
                'N_scale': float(sample['N_scale']),
                'num_nodes': int(sample['num_nodes'])
            }

        logger.info(f"Sampled {len(tests)} exact training data points with targets")

        return tests, training_targets

    def _extract_base_coprime_pairs(self) -> set:
        """
        Extract BASE coprime pairs from scaled training data.

        Training uses base coprime a∈[2,10], then scales by factors [1,9].
        This reverses that process to get the base pairs.

        Returns:
            Set of (a_base, b_base) tuples
        """
        from Generate_training_data import compute_sym_factor

        BASE_MIN_A = 2
        BASE_MAX_A = 10
        SCALE_FACTORS = range(1, 10)  # TRAINING_SAMPLE_MIN_FACTOR=1, MAX_FACTOR=10

        base_pairs = set()
        for (a_scaled, b_scaled) in self.parameter_stats['ab_pairs_dict'].keys():
            # Try each scale factor to find base coprime
            for factor in SCALE_FACTORS:
                if a_scaled % factor == 0 and b_scaled % factor == 0:
                    a_base = a_scaled // factor
                    b_base = b_scaled // factor
                    if BASE_MIN_A <= a_base <= BASE_MAX_A and np.gcd(a_base, b_base) == 1:
                        base_pairs.add((a_base, b_base))
                        # Also add symmetric conjugate
                        sym = compute_sym_factor(a_base, b_base)
                        base_pairs.add((sym[0], sym[1]))
                        break  # Found valid base pair

        logger.info(f"Extracted {len(base_pairs)} base coprime pairs from training")
        return base_pairs

    def _generate_weight_interpolation_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with weights BETWEEN training values (true interpolation).

        Uses base coprime pairs from training [2,10] with scaling [1,9].
        Thresholds from training ranges.
        Weights sampled between training min/max.
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(42)

        # Get base coprime pairs from training
        training_base_pairs = self._extract_base_coprime_pairs()
        base_pairs_list = list(training_base_pairs)

        for _ in range(num_tests):
            # Sample base coprime pair from training
            a_base, b_base = base_pairs_list[np.random.randint(0, len(base_pairs_list))]

            # Sample scaling factor [1,9]
            scale_factor = np.random.randint(1, 10)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # Sample thresholds from training ranges
            interlayer_thresh = np.random.uniform(
                ranges['interlayer_dist_threshold']['min'],
                ranges['interlayer_dist_threshold']['max']
            )
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['min'],
                ranges['intralayer_dist_threshold']['max']
            )

            # INTERPOLATE weights: sample BETWEEN min and max
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['min'],
                ranges['inter_graph_weight']['max']
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['min'],
                ranges['intra_graph_weight']['max']
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} weight interpolation tests")
        return tests

    def _generate_weight_extrapolation_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with weights ABOVE training maximum (extrapolation).

        Uses base coprime pairs from training [2,10] with scaling [1,9].
        Thresholds from training ranges.
        Weights sampled ABOVE training max.
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(43)

        # Get base coprime pairs from training
        training_base_pairs = self._extract_base_coprime_pairs()
        base_pairs_list = list(training_base_pairs)

        # Calculate extrapolation ranges (above max)
        inter_weight_range = ranges['inter_graph_weight']['max'] - ranges['inter_graph_weight']['min']
        intra_weight_range = ranges['intra_graph_weight']['max'] - ranges['intra_graph_weight']['min']

        for _ in range(num_tests):
            # Sample base coprime pair from training
            a_base, b_base = base_pairs_list[np.random.randint(0, len(base_pairs_list))]

            # Sample scaling factor [1,9]
            scale_factor = np.random.randint(1, 10)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # Sample thresholds from training ranges
            interlayer_thresh = np.random.uniform(
                ranges['interlayer_dist_threshold']['min'],
                ranges['interlayer_dist_threshold']['max']
            )
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['min'],
                ranges['intralayer_dist_threshold']['max']
            )

            # EXTRAPOLATE weights: sample ABOVE max (up to max + 20% of range)
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['max'],
                ranges['inter_graph_weight']['max'] + 0.2 * inter_weight_range
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['max'],
                ranges['intra_graph_weight']['max'] + 0.2 * intra_weight_range
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} weight extrapolation tests")
        return tests

    def _generate_threshold_extrapolation_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with thresholds ABOVE training maximum (extrapolation).

        Uses base coprime pairs from training [2,10] with scaling [1,9].
        Weights from training ranges.
        Both thresholds sampled ABOVE training max.
        ENFORCES: intralayer_thresh < interlayer_thresh (physics constraint).
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(44)

        # Get base coprime pairs from training
        training_base_pairs = self._extract_base_coprime_pairs()
        base_pairs_list = list(training_base_pairs)

        # Calculate extrapolation ranges (above max)
        interlayer_range = ranges['interlayer_dist_threshold']['max'] - ranges['interlayer_dist_threshold']['min']
        intralayer_range = ranges['intralayer_dist_threshold']['max'] - ranges['intralayer_dist_threshold']['min']

        for _ in range(num_tests):
            # Sample base coprime pair from training
            a_base, b_base = base_pairs_list[np.random.randint(0, len(base_pairs_list))]

            # Sample scaling factor [1,9]
            scale_factor = np.random.randint(1, 10)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # EXTRAPOLATE thresholds: sample ABOVE max (up to max + 20% of range)
            # Ensure intralayer < interlayer
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['max'],
                ranges['intralayer_dist_threshold']['max'] + 0.2 * intralayer_range
            )
            interlayer_thresh = np.random.uniform(
                max(intralayer_thresh + 0.01, ranges['interlayer_dist_threshold']['max']),
                ranges['interlayer_dist_threshold']['max'] + 0.2 * interlayer_range
            )

            # Sample weights from training ranges
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['min'],
                ranges['inter_graph_weight']['max']
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['min'],
                ranges['intra_graph_weight']['max']
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} threshold extrapolation tests")
        return tests

    def _generate_scaling_invariance_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with LARGE scaling factors [10,20] to test scale invariance.

        Uses base coprime pairs from training [2,10].
        Thresholds/weights from training ranges.
        Scaling factors [10,20] - network should predict same k-points as factors [1,9].
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(45)

        # Get base coprime pairs from training
        training_base_pairs = self._extract_base_coprime_pairs()
        base_pairs_list = list(training_base_pairs)

        for _ in range(num_tests):
            # Sample base coprime pair from training
            a_base, b_base = base_pairs_list[np.random.randint(0, len(base_pairs_list))]

            # Sample LARGE scaling factor [10,20]
            scale_factor = np.random.randint(10, 21)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # Sample thresholds/weights from training ranges
            interlayer_thresh = np.random.uniform(
                ranges['interlayer_dist_threshold']['min'],
                ranges['interlayer_dist_threshold']['max']
            )
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['min'],
                ranges['intralayer_dist_threshold']['max']
            )
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['min'],
                ranges['inter_graph_weight']['max']
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['min'],
                ranges['intra_graph_weight']['max']
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} scaling invariance tests (factors [10,20])")
        return tests

    def _generate_coprime_extrapolation_close_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with NEW base coprime pairs a∈[11,15] (close extrapolation).

        Samples coprime pairs OUTSIDE training base range [2,10].
        Checks symmetry to avoid duplicates.
        Size limit: N<500 to prevent huge systems.
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(46)

        from utils import compute_twist_constants
        from Generate_training_data import compute_sym_factor

        # Get training base pairs to check symmetry
        training_base_pairs = self._extract_base_coprime_pairs()

        # Extrapolation range
        EXTRAP_MIN_A = 11
        EXTRAP_MAX_A = 15

        extrapolation_pairs = []
        attempts = 0
        max_attempts = num_tests * 20

        while len(extrapolation_pairs) < num_tests and attempts < max_attempts:
            attempts += 1

            # Sample base coprime in extrapolation range
            a = np.random.randint(EXTRAP_MIN_A, EXTRAP_MAX_A + 1)
            b = np.random.randint(1, a)

            # Ensure coprime
            if np.gcd(a, b) != 1:
                continue

            pair = (a, b)
            sym_pair = tuple(compute_sym_factor(a, b))

            # Check not in training (including symmetry)
            if pair in training_base_pairs or sym_pair in training_base_pairs:
                continue

            # Check not already added (including symmetry)
            if pair in extrapolation_pairs or sym_pair in extrapolation_pairs:
                continue

            # Check system size (with typical scaling factor)
            try:
                N, _, _, _ = compute_twist_constants(a, b)
                # Estimate nodes with max scaling factor 9: N * 9^2 = N * 81
                if N * 81 > 500:
                    continue
                extrapolation_pairs.append(pair)
            except:
                continue

        logger.info(f"Generated {len(extrapolation_pairs)} close extrapolation coprime pairs (a∈[{EXTRAP_MIN_A},{EXTRAP_MAX_A}])")

        # Generate tests with these pairs
        for a_base, b_base in extrapolation_pairs[:num_tests]:
            # Sample scaling factor [1,9]
            scale_factor = np.random.randint(1, 10)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # Sample thresholds/weights from training ranges
            interlayer_thresh = np.random.uniform(
                ranges['interlayer_dist_threshold']['min'],
                ranges['interlayer_dist_threshold']['max']
            )
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['min'],
                ranges['intralayer_dist_threshold']['max']
            )
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['min'],
                ranges['inter_graph_weight']['max']
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['min'],
                ranges['intra_graph_weight']['max']
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} close coprime extrapolation tests")
        return tests

    def _generate_coprime_extrapolation_far_tests(self, num_tests: int) -> List[List[float]]:
        """
        Generate tests with NEW base coprime pairs a∈[16,20] (far extrapolation).

        Samples coprime pairs FAR outside training base range [2,10].
        Checks symmetry to avoid duplicates.
        Uses smaller scaling factors [1,5] to keep system size reasonable.
        Size limit: N<500 to prevent huge systems.
        """
        tests = []
        ranges = self.parameter_stats['parameter_ranges']
        np.random.seed(47)

        from utils import compute_twist_constants
        from Generate_training_data import compute_sym_factor

        # Get training base pairs to check symmetry
        training_base_pairs = self._extract_base_coprime_pairs()

        # Far extrapolation range
        EXTRAP_MIN_A = 16
        EXTRAP_MAX_A = 20

        extrapolation_pairs = []
        attempts = 0
        max_attempts = num_tests * 20

        while len(extrapolation_pairs) < num_tests and attempts < max_attempts:
            attempts += 1

            # Sample base coprime in far extrapolation range
            a = np.random.randint(EXTRAP_MIN_A, EXTRAP_MAX_A + 1)
            b = np.random.randint(1, a)

            # Ensure coprime
            if np.gcd(a, b) != 1:
                continue

            pair = (a, b)
            sym_pair = tuple(compute_sym_factor(a, b))

            # Check not in training (including symmetry)
            if pair in training_base_pairs or sym_pair in training_base_pairs:
                continue

            # Check not already added (including symmetry)
            if pair in extrapolation_pairs or sym_pair in extrapolation_pairs:
                continue

            # Check system size (with smaller scaling factors [1,5])
            try:
                N, _, _, _ = compute_twist_constants(a, b)
                # Estimate nodes with max scaling factor 5: N * 5^2 = N * 25
                if N * 25 > 500:
                    continue
                extrapolation_pairs.append(pair)
            except:
                continue

        logger.info(f"Generated {len(extrapolation_pairs)} far extrapolation coprime pairs (a∈[{EXTRAP_MIN_A},{EXTRAP_MAX_A}])")

        # Generate tests with these pairs
        for a_base, b_base in extrapolation_pairs[:num_tests]:
            # Sample SMALLER scaling factor [1,5] (keep system size reasonable)
            scale_factor = np.random.randint(1, 6)
            a_scaled = a_base * scale_factor
            b_scaled = b_base * scale_factor

            # Sample thresholds/weights from training ranges
            interlayer_thresh = np.random.uniform(
                ranges['interlayer_dist_threshold']['min'],
                ranges['interlayer_dist_threshold']['max']
            )
            intralayer_thresh = np.random.uniform(
                ranges['intralayer_dist_threshold']['min'],
                ranges['intralayer_dist_threshold']['max']
            )
            inter_weight = np.random.uniform(
                ranges['inter_graph_weight']['min'],
                ranges['inter_graph_weight']['max']
            )
            intra_weight = np.random.uniform(
                ranges['intra_graph_weight']['min'],
                ranges['intra_graph_weight']['max']
            )

            test_params = [
                int(a_scaled), int(b_scaled),
                float(interlayer_thresh),
                float(intralayer_thresh),
                float(inter_weight),
                float(intra_weight)
            ]
            tests.append(test_params)

        logger.info(f"Generated {len(tests)} far coprime extrapolation tests")
        return tests

    def _generate_redundancy_tests(self, num_pairs: int) -> List[List[List[float]]]:
        """
        Generate scale-equivalent system pairs (e.g., [7,6] and [14,12]).

        IMPORTANT: These pairs give the SAME graph size and should produce IDENTICAL physics!
        This tests if the network learned true physical invariance.
        """
        redundancy_pairs = []

        # Get scale-equivalent groups
        equiv_groups = self.parameter_stats['scale_equivalent_groups']

        if not equiv_groups:
            logger.warning("No scale-equivalent groups found in training data")
            return []

        # Sample from available equivalence groups
        np.random.seed(44)

        ranges = self.parameter_stats['parameter_ranges']

        # Get groups with at least 2 pairs
        valid_groups = [(ratio, pairs) for ratio, pairs in equiv_groups.items() if len(pairs) >= 2]

        for _ in range(min(num_pairs, len(valid_groups))):
            # Pick a random equivalence group
            group_idx = np.random.randint(0, len(valid_groups))
            ratio, ab_options = valid_groups[group_idx]

            if len(ab_options) >= 2:
                # Pick two different (a,b) pairs with same ratio (same physics!)
                pair_indices = np.random.choice(len(ab_options), size=2, replace=False)
                ab_pair_1 = ab_options[pair_indices[0]]
                ab_pair_2 = ab_options[pair_indices[1]]

                # Use SAME other parameters for both (to test physical equivalence)
                interlayer_thresh = np.random.uniform(
                    ranges['interlayer_dist_threshold']['min'],
                    ranges['interlayer_dist_threshold']['max']
                )
                intralayer_thresh = np.random.uniform(
                    ranges['intralayer_dist_threshold']['min'],
                    ranges['intralayer_dist_threshold']['max']
                )
                inter_weight = np.random.uniform(
                    ranges['inter_graph_weight']['min'],
                    ranges['inter_graph_weight']['max']
                )
                intra_weight = np.random.uniform(
                    ranges['intra_graph_weight']['min'],
                    ranges['intra_graph_weight']['max']
                )

                params_1 = [
                    int(ab_pair_1[0]), int(ab_pair_1[1]),
                    float(interlayer_thresh), float(intralayer_thresh),
                    float(inter_weight), float(intra_weight)
                ]
                params_2 = [
                    int(ab_pair_2[0]), int(ab_pair_2[1]),
                    float(interlayer_thresh), float(intralayer_thresh),
                    float(inter_weight), float(intra_weight)
                ]

                redundancy_pairs.append([params_1, params_2])

        return redundancy_pairs

    def save_benchmark_suite(self, benchmark_suite: Dict[str, Any], filename: str = None) -> None:
        """
        Save benchmark suite to file for reproducibility.

        Args:
            benchmark_suite: Generated benchmark test suite
            filename: Output filename (JSON format)
        """
        import json

        if filename is None:
            filename = os.path.join(constants.PATH, "benchmark_test_suite.json")

        # Convert numpy types to native Python for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj

        serializable_suite = convert_to_serializable(benchmark_suite)

        with open(filename, 'w') as f:
            json.dump(serializable_suite, f, indent=2)

        logger.info(f"Benchmark suite saved to {filename}")

    def print_summary(self, benchmark_suite: Dict[str, Any]) -> None:
        """Print human-readable summary of benchmark suite."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK SUITE SUMMARY")
        print("=" * 80)

        print(f"\nTraining Data: {self.parameter_stats['total_samples']:,} samples")
        print(f"Unique (a,b) pairs in training: {self.parameter_stats['unique_ab_pairs']}")
        print(f"Scale-equivalent groups: {self.parameter_stats['num_scale_groups']}")

        print("\n--- PARAMETER RANGES IN TRAINING DATA ---")
        for param, stats in self.parameter_stats['parameter_ranges'].items():
            print(f"{param:30s}: [{stats['min']:8.3f}, {stats['max']:8.3f}]  "
                  f"({stats['unique_values']} unique values)")

        print("\n--- BENCHMARK TEST CATEGORIES ---")
        print(f"0. Training samples:                {len(benchmark_suite['training_samples']):4d}  (baseline)")
        print(f"1. Weight interpolation:            {len(benchmark_suite['weight_interpolation']):4d}  (weights between training min/max)")
        print(f"2. Weight extrapolation:            {len(benchmark_suite['weight_extrapolation']):4d}  (weights above training max)")
        print(f"3. Threshold extrapolation:         {len(benchmark_suite['threshold_extrapolation']):4d}  (thresholds above training max)")
        print(f"4. Scaling invariance:              {len(benchmark_suite['scaling_invariance']):4d}  (scale factors [10,20])")
        print(f"5. Coprime extrap (close):          {len(benchmark_suite['coprime_extrapolation_close']):4d}  (new coprime a∈[11,15])")
        print(f"6. Coprime extrap (far):            {len(benchmark_suite['coprime_extrapolation_far']):4d}  (new coprime a∈[16,20])")
        print(f"7. Redundancy pairs:                {len(benchmark_suite['redundancy_pairs']):4d} pairs  (scale-equivalent systems)")

        total_individual_tests = (len(benchmark_suite['training_samples']) +
                                 len(benchmark_suite['weight_interpolation']) +
                                 len(benchmark_suite['weight_extrapolation']) +
                                 len(benchmark_suite['threshold_extrapolation']) +
                                 len(benchmark_suite['scaling_invariance']) +
                                 len(benchmark_suite['coprime_extrapolation_close']) +
                                 len(benchmark_suite['coprime_extrapolation_far']) +
                                 sum(len(pair) for pair in benchmark_suite['redundancy_pairs']))

        print(f"\nTOTAL PARAMETER SETS: {total_individual_tests}")

        print("\n--- SAMPLE WEIGHT INTERPOLATION TESTS ---")
        for i, params in enumerate(benchmark_suite['weight_interpolation'][:3], 1):
            print(f"  {i}. a={params[0]}, b={params[1]}, "
                  f"inter_th={params[2]:.2f}, intra_th={params[3]:.2f}, "
                  f"inter_w={params[4]:.2f}, intra_w={params[5]:.2f}")

        if benchmark_suite['redundancy_pairs']:
            print("\n--- SAMPLE REDUNDANCY PAIRS (Should give IDENTICAL physics) ---")
            for i, pair in enumerate(benchmark_suite['redundancy_pairs'][:3], 1):
                p1, p2 = pair
                print(f"  Pair {i}: (SAME graph size, ratio={p1[0]/p1[1]:.3f})")
                print(f"    System 1: a={p1[0]:2d}, b={p1[1]:2d}")
                print(f"    System 2: a={p2[0]:2d}, b={p2[1]:2d}")
                print(f"    Parameters: inter_th={p1[2]:.2f}, intra_th={p1[3]:.2f}, "
                      f"inter_w={p1[4]:.2f}, intra_w={p1[5]:.2f}")

        print("\n" + "=" * 80)


def main():
    """Generate comprehensive benchmark suite."""

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE BENCHMARK SUITE GENERATION")
    logger.info("=" * 80)

    # Create benchmark suite designer
    designer = benchmark_test_suite()

    # Analyze training data
    logger.info("\n1. Analyzing training data...")
    stats = designer.analyze_training_data()

    # Generate comprehensive benchmark suite (uses constants by default)
    logger.info("\n2. Generating benchmark suite...")
    suite = designer.generate_benchmark_suite(
        # Uses constants.BENCHMARK_NUM_REDUNDANCY_TESTS by default
    )

    # Print summary
    designer.print_summary(suite)

    # Save to file
    logger.info("\n3. Saving benchmark suite...")
    designer.save_benchmark_suite(suite)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUITE GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Run baseline benchmark on data-trained model")
    logger.info("  2. Apply physics-based training")
    logger.info("  3. Run benchmark again to measure improvements")
    logger.info("  4. Compare results to quantify physics training benefits")


if __name__ == "__main__":
    main()
