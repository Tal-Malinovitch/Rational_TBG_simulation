"""
Report generation for Dirac point neural network benchmarks.

This module provides comprehensive report generation capabilities for benchmark
results, including JSON/dict reports, detailed text logs, and CSV exports.

Classes:
    dirac_network_report_generator: Report generation and formatting
"""

import constants
from constants import np, logging, os, time
from constants import List, Dict, Tuple, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)


class dirac_network_report_generator:
    """
    Provides comprehensive report generation for benchmark results.

    This class generates various report formats including JSON/dict summaries,
    detailed text logs, and CSV exports of benchmark timing and accuracy data.
    """

    def __init__(self) -> None:
        """Initialize the report generator."""
        logger.info("dirac_network_report_generator initialized")

    def generate_performance_report(
        self,
        nn_prediction_times: List[float],
        physics_computation_times: List[float],
        benchmark_config: dict,
        accuracy_comparison: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis report.

        Args:
            nn_prediction_times (List[float]): NN prediction times in seconds
            physics_computation_times (List[float]): Physics computation times in seconds
            benchmark_config (dict): Benchmark configuration parameters
            accuracy_comparison (Dict, optional): Accuracy analysis results

        Returns:
            Dict[str, Any]: Detailed performance report
        """
        if not nn_prediction_times and not physics_computation_times:
            return {"error": "No benchmark data available. Run benchmarks first."}

        report = {
            'benchmark_summary': {
                'nn_predictions_count': len(nn_prediction_times),
                'physics_computations_count': len(physics_computation_times),
                'benchmark_config': benchmark_config.copy()
            }
        }

        # NN prediction statistics
        if nn_prediction_times:
            nn_times_ms = [t * 1000 for t in nn_prediction_times]
            report['nn_performance'] = {
                'avg_time_ms': np.mean(nn_times_ms),
                'min_time_ms': np.min(nn_times_ms),
                'max_time_ms': np.max(nn_times_ms),
                'std_time_ms': np.std(nn_times_ms),
                'median_time_ms': np.median(nn_times_ms),
                'percentile_95_ms': np.percentile(nn_times_ms, 95)
            }

        # Physics computation statistics
        if physics_computation_times:
            physics_times_ms = [t * 1000 for t in physics_computation_times]
            report['physics_performance'] = {
                'avg_time_ms': np.mean(physics_times_ms),
                'min_time_ms': np.min(physics_times_ms),
                'max_time_ms': np.max(physics_times_ms),
                'std_time_ms': np.std(physics_times_ms),
                'median_time_ms': np.median(physics_times_ms),
                'percentile_95_ms': np.percentile(physics_times_ms, 95)
            }

        # Overall acceleration analysis
        if nn_prediction_times and physics_computation_times:
            min_len = min(len(nn_prediction_times), len(physics_computation_times))
            acceleration_factors = [
                physics_computation_times[i] / nn_prediction_times[i]
                for i in range(min_len) if nn_prediction_times[i] > 0
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

        # Include accuracy comparison if provided
        if accuracy_comparison:
            report['accuracy_comparison'] = accuracy_comparison

        return report

    def save_detailed_benchmark_log(
        self,
        filename: str,
        report: Dict[str, Any],
        nn_prediction_times: List[float],
        physics_computation_times: List[float]
    ) -> None:
        """
        Save detailed benchmark statistics to a comprehensive text log file.

        Args:
            filename (str): Path to output log file
            report (dict): Performance report from generate_performance_report()
            nn_prediction_times (List[float]): All NN prediction times
            physics_computation_times (List[float]): All physics computation times
        """
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
                self._write_accuracy_section(f, report['accuracy_comparison'])

            # Raw Data Summary
            f.write("RAW TIMING DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total NN Predictions:        {len(nn_prediction_times)}\n")
            f.write(f"  Total Physics Computations:  {len(physics_computation_times)}\n")
            if nn_prediction_times:
                f.write(f"\n  All NN Prediction Times (ms):\n")
                for i, t in enumerate(nn_prediction_times, 1):
                    f.write(f"    {i:4d}: {t*1000:.6f}\n")
            if physics_computation_times:
                f.write(f"\n  All Physics Computation Times (ms):\n")
                for i, t in enumerate(physics_computation_times, 1):
                    f.write(f"    {i:4d}: {t*1000:.6f}\n")

            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF BENCHMARK ANALYSIS\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Detailed benchmark log written to {filename}")

    def _write_accuracy_section(self, f, accuracy: Dict[str, Any]) -> None:
        """
        Write accuracy comparison section to file.

        Args:
            f: File handle
            accuracy: Accuracy comparison dict
        """
        f.write("PREDICTION ACCURACY ANALYSIS\n")
        f.write("=" * 80 + "\n")

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

        # K-space Separation Statistics (for two-point predictions)
        if 'k_separation' in accuracy:
            f.write("  K-space Separation Between Predictions:\n")
            f.write(f"    {accuracy['k_separation'].get('description', '')}\n")
            k_sep = accuracy['k_separation']
            f.write(f"    Mean:               {k_sep.get('mean', 0):.6f}\n")
            f.write(f"    Median:             {k_sep.get('median', 0):.6f}\n")
            f.write(f"    Std Deviation:      {k_sep.get('std', 0):.6f}\n")
            f.write(f"    Min:                {k_sep.get('min', 0):.6f}\n")
            f.write(f"    Max:                {k_sep.get('max', 0):.6f}\n")
            f.write(f"    25th Percentile:    {k_sep.get('percentile_25', 0):.6f}\n")
            f.write(f"    75th Percentile:    {k_sep.get('percentile_75', 0):.6f}\n")
            f.write(f"    Num Measurements:   {k_sep.get('num_measurements', 0)}\n\n")

        # Detailed Per-Test Comparison
        if 'comparison_details' in accuracy:
            f.write("\nDETAILED PER-TEST COMPARISON\n")
            f.write("-" * 80 + "\n")
            for i, detail in enumerate(accuracy['comparison_details'], 1):
                # Check if this is a two-point prediction
                pred_idx = detail.get('prediction_index', 0)
                pred_label = ""
                if pred_idx == 1:
                    pred_label = " [Prediction 1]"
                elif pred_idx == 2:
                    pred_label = " [Prediction 2]"

                f.write(f"\n  Test {i}{pred_label}: Parameters {detail.get('params', [])}\n")

                # Show k-space separation for two-point predictions
                if detail.get('k_separation') is not None and detail.get('k_separation') > 0:
                    f.write(f"    K-space Separation: {detail.get('k_separation', 0):.6f}\n")

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
