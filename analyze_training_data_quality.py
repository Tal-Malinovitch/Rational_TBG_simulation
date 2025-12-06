"""
Analyze quality of training data Dirac points.

This script:
1. Loads all training data points
2. For each coprime (a,b) parameter set, builds TBG system
3. Evaluates Dirac point quality metrics at each training point
4. Reports statistics on how many points pass/fail quality criteria

Quality criteria used:
- Training generation: weighted metric < threshold (from Generate_training_data.py)
- Benchmark phase: gap < 0.001, R2 < 0.1 (from dirac_network_benchmark.py)
"""

import constants
from constants import np, logging, csv, os
from constants import List, Tuple, Dict, DefaultDict, Any
from collections import defaultdict
from TBG import Dirac_analysis
from dirac_network_builder import dirac_network_builder

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def analyze_training_data_quality():
    """Analyze quality of all coprime training data points."""

    # Load training data
    logger.info("Loading training data...")
    print("="*80)
    print("Training Data Quality Analysis")
    print("="*80)
    print()

    data_file = 'Training_data/dirac_training_data.csv'

    # Group by parameter set
    param_points: DefaultDict[Tuple, List[Tuple[float, float, float]]] = defaultdict(list)

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)

        total_rows = 0
        coprime_rows = 0

        for row in reader:
            total_rows += 1

            a = int(row['a'])
            b = int(row['b'])

            # Only process coprime pairs
            if np.gcd(a, b) != 1:
                continue

            coprime_rows += 1

            param_key = (
                a, b,
                float(row['interlayer_dist_threshold']),
                float(row['intralayer_dist_threshold']),
                float(row['inter_graph_weight']),
                float(row['intra_graph_weight'])
            )

            k_x = float(row['target_k_x'])
            k_y = float(row['target_k_y'])
            velocity = float(row['Dirac_velocity'])

            param_points[param_key].append((k_x, k_y, velocity))

    print(f"Loaded {total_rows:,} total rows")
    print(f"Coprime (a,b) pairs: {coprime_rows:,} rows")
    print(f"Unique coprime parameter sets: {len(param_points):,}")
    print()

    # Quality criteria
    weights = constants.DEFAULT_NN_LOSS_WEIGHTS  # [0.6, 0.3, 0.1] for [gap, R2, isotropy]

    # Thresholds
    TRAINING_WEIGHTED_THRESHOLD = 0.05  # From Generate_training_data
    BENCHMARK_GAP_THRESHOLD = 0.001
    BENCHMARK_R2_THRESHOLD = 0.1

    print("Quality Criteria:")
    print(f"  Training generation: weighted_metric < {TRAINING_WEIGHTED_THRESHOLD}")
    print(f"    (weighted = {weights[0]}*gap + {weights[1]}*R2 + {weights[2]}*isotropy)")
    print(f"  Benchmark phase: gap < {BENCHMARK_GAP_THRESHOLD} AND R2 < {BENCHMARK_R2_THRESHOLD}")
    print()

    # Statistics
    total_points = 0
    pass_training_criteria = 0
    pass_benchmark_criteria = 0
    pass_both_criteria = 0

    gap_values = []
    r2_values = []
    isotropy_values = []
    weighted_metrics = []

    # Process ALL parameter sets
    param_items = list(param_points.items())

    print(f"Processing ALL {len(param_items)} parameter sets")
    print("This will take a while...")
    print()

    processed = 0
    for param_key, points in param_items:
        a, b, inter_th, intra_th, inter_w, intra_w = param_key

        # Build TBG system
        params = [a, b, inter_th, intra_th, inter_w, intra_w]

        try:
            builder = dirac_network_builder()
            builder.build_network()
            builder.set_network_parameters(params)
            pg = builder.current_periodic_graph

            if pg is None:
                logger.warning(f"Failed to build graph for a={a}, b={b}")
                continue

            dirac_analyzer = Dirac_analysis(pg)

            # Check each point
            for k_x, k_y, velocity in points:
                total_points += 1

                try:
                    metrics, _, _, _, _ = dirac_analyzer.check_Dirac_point((k_x, k_y), 1)
                    gap, R2, isotropy = metrics

                    weighted_metric = weights[0]*gap + weights[1]*R2 + weights[2]*isotropy

                    # Record statistics
                    gap_values.append(gap)
                    r2_values.append(R2)
                    isotropy_values.append(isotropy)
                    weighted_metrics.append(weighted_metric)

                    # Check criteria
                    passes_training = weighted_metric < TRAINING_WEIGHTED_THRESHOLD
                    passes_benchmark = gap < BENCHMARK_GAP_THRESHOLD and R2 < BENCHMARK_R2_THRESHOLD

                    if passes_training:
                        pass_training_criteria += 1
                    if passes_benchmark:
                        pass_benchmark_criteria += 1
                    if passes_training and passes_benchmark:
                        pass_both_criteria += 1

                except Exception as e:
                    logger.warning(f"Failed to check point k=({k_x:.4f}, {k_y:.4f}): {e}")

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(param_items)} parameter sets ({total_points} points analyzed so far)...", flush=True)

        except Exception as e:
            logger.warning(f"Failed to process a={a}, b={b}: {e}")
            continue

    print(f"  Processed {processed}/{len(param_items)} parameter sets")
    print(f"  Total points analyzed: {total_points}")
    print()

    # Report results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    print(f"Total coprime data points analyzed: {total_points:,}")
    print()

    print("Quality Statistics:")
    print(f"  Pass training criteria (weighted < {TRAINING_WEIGHTED_THRESHOLD}): {pass_training_criteria:,} ({100*pass_training_criteria/total_points:.1f}%)")
    print(f"  Pass benchmark criteria (gap < {BENCHMARK_GAP_THRESHOLD}, R2 < {BENCHMARK_R2_THRESHOLD}): {pass_benchmark_criteria:,} ({100*pass_benchmark_criteria/total_points:.1f}%)")
    print(f"  Pass BOTH criteria: {pass_both_criteria:,} ({100*pass_both_criteria/total_points:.1f}%)")
    print()

    print(f"  FAIL training criteria: {total_points - pass_training_criteria:,} ({100*(total_points - pass_training_criteria)/total_points:.1f}%)")
    print(f"  FAIL benchmark criteria: {total_points - pass_benchmark_criteria:,} ({100*(total_points - pass_benchmark_criteria)/total_points:.1f}%)")
    print()

    # Metric distributions
    print("Metric Distributions:")
    print(f"  Gap:")
    print(f"    Mean: {np.mean(gap_values):.6f}, Median: {np.median(gap_values):.6f}")
    print(f"    Min: {np.min(gap_values):.6f}, Max: {np.max(gap_values):.6f}")
    print(f"    Std: {np.std(gap_values):.6f}")
    print(f"    P25: {np.percentile(gap_values, 25):.6f}, P75: {np.percentile(gap_values, 75):.6f}")
    print()

    print(f"  R2:")
    print(f"    Mean: {np.mean(r2_values):.6f}, Median: {np.median(r2_values):.6f}")
    print(f"    Min: {np.min(r2_values):.6f}, Max: {np.max(r2_values):.6f}")
    print(f"    Std: {np.std(r2_values):.6f}")
    print(f"    P25: {np.percentile(r2_values, 25):.6f}, P75: {np.percentile(r2_values, 75):.6f}")
    print()

    print(f"  Isotropy:")
    print(f"    Mean: {np.mean(isotropy_values):.6f}, Median: {np.median(isotropy_values):.6f}")
    print(f"    Min: {np.min(isotropy_values):.6f}, Max: {np.max(isotropy_values):.6f}")
    print(f"    Std: {np.std(isotropy_values):.6f}")
    print(f"    P25: {np.percentile(isotropy_values, 25):.6f}, P75: {np.percentile(isotropy_values, 75):.6f}")
    print()

    print(f"  Weighted Metric:")
    print(f"    Mean: {np.mean(weighted_metrics):.6f}, Median: {np.median(weighted_metrics):.6f}")
    print(f"    Min: {np.min(weighted_metrics):.6f}, Max: {np.max(weighted_metrics):.6f}")
    print(f"    Std: {np.std(weighted_metrics):.6f}")
    print(f"    P25: {np.percentile(weighted_metrics, 25):.6f}, P75: {np.percentile(weighted_metrics, 75):.6f}")
    print()

    print("="*80)

if __name__ == "__main__":
    analyze_training_data_quality()
