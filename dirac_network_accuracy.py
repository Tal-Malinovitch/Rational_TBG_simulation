"""
Accuracy analysis for Dirac point neural network predictions.

This module provides comprehensive accuracy analysis capabilities for comparing
neural network predictions against physics computations, including error metrics,
statistical analysis, and binned error analysis.

Classes:
    dirac_network_accuracy_analyzer: Accuracy comparison and error analysis
"""

import constants
from constants import np, logging
from constants import List, Dict, Tuple, Optional, Union, Any
from TBG import Dirac_analysis

# Configure logging
logger = logging.getLogger(__name__)


class dirac_network_accuracy_analyzer:
    """
    Provides comprehensive accuracy analysis for Dirac point predictions.

    This class compares neural network predictions against physics computations,
    calculating various error metrics and providing detailed statistical analysis.

    Attributes:
        network_builder (Optional): Network builder for physics loss calculations
    """

    def __init__(self, network_builder=None) -> None:
        """
        Initialize the accuracy analyzer.

        Args:
            network_builder (optional): Network builder for accessing physics graphs
        """
        self.network_builder = network_builder
        logger.info("dirac_network_accuracy_analyzer initialized")

    def set_network_builder(self, network_builder) -> None:
        """
        Set the network builder for physics loss calculations.

        Args:
            network_builder: Network builder with current_periodic_graph
        """
        self.network_builder = network_builder
        logger.debug("Network builder set for accuracy analysis")

    def load_all_training_targets_for_params(self, params: List[float]) -> List[Dict[str, float]]:
        """
        Load ALL Dirac points from training data with the same parameters.

        For training_samples tests, we need to compare against the CLOSEST Dirac point,
        not just a single arbitrary one, because there are multiple Dirac points per parameter set.

        Args:
            params: [a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight]

        Returns:
            List of dicts with {target_k_x, target_k_y, target_velocity}
        """
        import csv
        import os

        training_data_path = os.path.join(constants.PATH, "Training_data", "dirac_training_data.csv")
        if not os.path.exists(training_data_path):
            logger.error(f"Training data not found: {training_data_path}")
            return []

        a, b = int(params[0]), int(params[1])
        # Rest of params with tolerance for floating point comparison
        interlayer_thresh = round(float(params[2]), 2)
        intralayer_thresh = round(float(params[3]), 2)
        inter_weight = round(float(params[4]), 2)
        intra_weight = round(float(params[5]), 2)

        matching_targets = []

        try:
            with open(training_data_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if parameters match
                    if (int(row['a']) == a and
                        int(row['b']) == b and
                        round(float(row['interlayer_dist_threshold']), 2) == interlayer_thresh and
                        round(float(row['intralayer_dist_threshold']), 2) == intralayer_thresh and
                        round(float(row['inter_graph_weight']), 2) == inter_weight and
                        round(float(row['intra_graph_weight']), 2) == intra_weight):

                        matching_targets.append({
                            'target_k_x': float(row['target_k_x']),
                            'target_k_y': float(row['target_k_y']),
                            'target_velocity': float(row['Dirac_velocity'])
                        })

            logger.debug(f"Found {len(matching_targets)} Dirac points for params {params[:2]}")
            return matching_targets

        except Exception as e:
            logger.error(f"Failed to load training targets: {e}")
            return []

    def find_closest_training_target(self, nn_k: Tuple[float, float],
                                    training_targets: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Find the closest training target Dirac point to the NN's prediction.

        Args:
            nn_k: (k_x, k_y) from NN prediction
            training_targets: List of training target dicts

        Returns:
            Closest target dict or None
        """
        if not training_targets:
            return None

        min_distance = float('inf')
        closest_target = None

        for target in training_targets:
            distance = np.sqrt((nn_k[0] - target['target_k_x'])**2 +
                             (nn_k[1] - target['target_k_y'])**2)
            if distance < min_distance:
                min_distance = distance
                closest_target = target

        logger.debug(f"Closest target at distance {min_distance:.6f}")
        return closest_target

    def compare_prediction_accuracy_against_training(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare NN predictions against TRAINING DATA targets (for training_samples tests).

        For each NN prediction, loads ALL Dirac points with the same parameters from training data
        and compares against the CLOSEST one.

        Args:
            detailed_results: Results from benchmark runs, must include 'params'

        Returns:
            Dict with accuracy statistics comparing NN vs closest training targets
        """
        if not detailed_results:
            return {'error': 'No results provided'}

        k_errors_to_closest = []
        v_errors_to_closest = []
        closest_targets_info = []

        for result in detailed_results:
            try:
                params = result.get('params', [])
                nn_pred = result.get('nn_prediction', [])

                if len(params) < 6 or len(nn_pred) < 6:
                    continue

                # Load all training targets for these parameters
                all_targets = self.load_all_training_targets_for_params(params)

                if not all_targets:
                    logger.warning(f"No training targets found for params {params[:2]}")
                    continue

                # Compare both NN predictions against closest targets
                for pred_idx in [0, 3]:  # k1,k2
                    nn_k = (nn_pred[pred_idx], nn_pred[pred_idx + 1])
                    nn_nu = nn_pred[pred_idx + 2]

                    # Find closest training target
                    closest = self.find_closest_training_target(nn_k, all_targets)

                    if closest:
                        # Calculate errors vs closest target
                        k_error = np.sqrt((nn_k[0] - closest['target_k_x'])**2 +
                                        (nn_k[1] - closest['target_k_y'])**2)

                        nn_nu_clamped = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, nn_nu))
                        nn_velocity = (1.0 - nn_nu_clamped) / nn_nu_clamped
                        v_error = abs(nn_velocity - closest['target_velocity'])

                        k_errors_to_closest.append(k_error)
                        v_errors_to_closest.append(v_error)

                        closest_targets_info.append({
                            'params': params,
                            'nn_k': list(nn_k),
                            'nn_velocity': nn_velocity,
                            'closest_target_k': [closest['target_k_x'], closest['target_k_y']],
                            'closest_target_v': closest['target_velocity'],
                            'k_error': k_error,
                            'v_error': v_error,
                            'num_targets_available': len(all_targets)
                        })

            except Exception as e:
                logger.warning(f"Failed to compare result: {e}")
                continue

        if not k_errors_to_closest:
            return {'error': 'No valid comparisons'}

        return {
            'comparison_type': 'training_targets',
            'k_error_to_closest_mean': np.mean(k_errors_to_closest),
            'k_error_to_closest_median': np.median(k_errors_to_closest),
            'k_error_to_closest_std': np.std(k_errors_to_closest),
            'v_error_to_closest_mean': np.mean(v_errors_to_closest),
            'v_error_to_closest_median': np.median(v_errors_to_closest),
            'v_error_to_closest_std': np.std(v_errors_to_closest),
            'num_comparisons': len(k_errors_to_closest),
            'closest_targets_details': closest_targets_info
        }

    def compare_prediction_accuracy(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                # NN prediction: [k_x1, k_y1, nu1, k_x2, k_y2, nu2] (two-point) or [k_x, k_y, nu] (single-point)
                nn_pred = result.get('nn_prediction', [])

                # Check if we have two-point predictions (new format)
                phys_result_1 = result.get('physics_result_1', None)
                phys_result_2 = result.get('physics_result_2', None)

                # Handle both two-point and single-point formats
                if phys_result_1 is not None and phys_result_2 is not None and len(nn_pred) >= 6:
                    # TWO-POINT FORMAT: Compare pred1 to physics1, pred2 to physics2
                    self._compare_single_pair(
                        nn_pred[0:3], phys_result_1, result, comparison_details,
                        k_x_errors, k_y_errors, k_magnitude_errors,
                        velocity_errors, velocity_relative_errors, log_velocity_errors,
                        nu_errors, nu_relative_errors, prediction_index=1
                    )
                    self._compare_single_pair(
                        nn_pred[3:6], phys_result_2, result, comparison_details,
                        k_x_errors, k_y_errors, k_magnitude_errors,
                        velocity_errors, velocity_relative_errors, log_velocity_errors,
                        nu_errors, nu_relative_errors, prediction_index=2
                    )

                    # Add k-space separation metric
                    k_separation = result.get('k_separation', 0.0)
                    if comparison_details:
                        comparison_details[-2]['k_separation'] = k_separation
                        comparison_details[-1]['k_separation'] = k_separation

                elif len(nn_pred) >= 3:
                    # SINGLE-POINT FORMAT (backward compatibility): Use old field name
                    phys_result = result.get('physics_result', [])
                    if len(phys_result) >= 4:
                        self._compare_single_pair(
                            nn_pred[0:3], phys_result, result, comparison_details,
                            k_x_errors, k_y_errors, k_magnitude_errors,
                            velocity_errors, velocity_relative_errors, log_velocity_errors,
                            nu_errors, nu_relative_errors, prediction_index=0
                        )

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

        # K-space separation statistics (for two-point predictions)
        k_separations = [d['k_separation'] for d in comparison_details if d.get('k_separation') is not None and d.get('k_separation') > 0]
        if k_separations:
            accuracy_metrics['k_separation'] = {
                'mean': np.mean(k_separations),
                'median': np.median(k_separations),
                'std': np.std(k_separations),
                'min': np.min(k_separations),
                'max': np.max(k_separations),
                'percentile_25': np.percentile(k_separations, 25),
                'percentile_75': np.percentile(k_separations, 75),
                'num_measurements': len(k_separations),
                'description': 'Euclidean distance between two predicted Dirac points in k-space (two-point predictions only)'
            }

        # Binned error analysis - group by system size and velocity magnitude
        binned_analysis = self.compute_binned_error_analysis(comparison_details)
        if binned_analysis:
            accuracy_metrics['binned_analysis'] = binned_analysis

        return accuracy_metrics

    def _compare_single_pair(
        self,
        nn_pred: List[float],
        phys_result: List[float],
        result: Dict[str, Any],
        comparison_details: List[Dict[str, Any]],
        k_x_errors: List[float],
        k_y_errors: List[float],
        k_magnitude_errors: List[float],
        velocity_errors: List[float],
        velocity_relative_errors: List[float],
        log_velocity_errors: List[float],
        nu_errors: List[float],
        nu_relative_errors: List[float],
        prediction_index: int = 0
    ) -> None:
        """
        Compare a single NN prediction to ADAM-converged physics result.

        Measures two quality metrics:
        1. k-distance: Distance from NN prediction to ADAM-converged Dirac point
        2. Physics loss: Quality metrics (gap, R², isotropy) at NN's predicted point

        Args:
            nn_pred: [k_x, k_y, nu] for this prediction
            phys_result: [converged_k_x, converged_k_y, converged_velocity] from ADAM optimization
            result: Dict with params and NN point quality metrics (nn_point_1_loss, etc)
            comparison_details: List to append detailed comparison to
            k_x_errors, k_y_errors, etc: Error lists to append to
            prediction_index: 0=single-point, 1=first pred, 2=second pred
        """
        # Extract k-points
        nn_k_x = nn_pred[0]
        nn_k_y = nn_pred[1]
        nn_nu = nn_pred[2]

        # NEW FORMAT: phys_result = [converged_k_x, converged_k_y, converged_velocity] from ADAM optimization
        phys_k_x = phys_result[0]  # Converged k_x from ADAM
        phys_k_y = phys_result[1]  # Converged k_y from ADAM
        phys_velocity = phys_result[2]  # Velocity at converged point

        # Convert NN's nu back to velocity: v = (1-nu)/nu
        nn_nu_clamped = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, nn_nu))
        nn_velocity = (1.0 - nn_nu_clamped) / nn_nu_clamped

        # Calculate errors
        k_x_error = abs(nn_k_x - phys_k_x)
        k_y_error = abs(nn_k_y - phys_k_y)
        k_magnitude_error = np.sqrt((nn_k_x - phys_k_x)**2 + (nn_k_y - phys_k_y)**2)
        velocity_error = abs(nn_velocity - phys_velocity)

        # Velocity error metrics (multiple for different perspectives)
        # 1. Standard relative error
        velocity_relative_error = velocity_error / abs(phys_velocity) if abs(phys_velocity) > 1e-10 else 0

        # 2. LOG-SCALE ERROR
        epsilon = 1e-6
        log_velocity_error = abs(np.log(abs(nn_velocity) + epsilon) - np.log(abs(phys_velocity) + epsilon))

        # 3. NU-SPACE ERROR
        phys_nu = 1.0 / (1.0 + abs(phys_velocity))
        nu_error = abs(nn_nu - phys_nu)
        nu_relative_error = nu_error / abs(phys_nu) if abs(phys_nu) > 1e-10 else 0

        # Append to error lists
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
        loss_absolute_diff = None

        if self.network_builder and self.network_builder.current_periodic_graph:
            try:
                # Calculate physics loss at NN predicted point
                dirac_analyzer = Dirac_analysis(self.network_builder.current_periodic_graph)
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
                loss_absolute_diff = nn_physics_loss - physics_physics_loss

                # Only compute ratio if physics loss is significant (> 1e-6)
                if physics_physics_loss > 1e-6:
                    loss_ratio = nn_physics_loss / physics_physics_loss
                else:
                    loss_ratio = None

            except Exception as e:
                logger.warning(f"Failed to compute physics loss comparison: {e}")

        # Extract NN point quality metrics from result dict
        if prediction_index == 1:
            nn_point_loss = result.get('nn_point_1_loss')
            nn_point_gap = result.get('nn_point_1_gap')
            nn_point_r2 = result.get('nn_point_1_r2')
            nn_point_isotropy = result.get('nn_point_1_isotropy')
        elif prediction_index == 2:
            nn_point_loss = result.get('nn_point_2_loss')
            nn_point_gap = result.get('nn_point_2_gap')
            nn_point_r2 = result.get('nn_point_2_r2')
            nn_point_isotropy = result.get('nn_point_2_isotropy')
        else:
            # Single point case - metrics might not be in result dict
            nn_point_loss = None
            nn_point_gap = None
            nn_point_r2 = None
            nn_point_isotropy = None

        # Build detailed comparison entry
        detail_entry = {
            'params': result.get('params', []),
            'prediction_index': prediction_index,  # 0=single, 1=pred1, 2=pred2
            # Raw values
            'nn_k_x': nn_k_x,
            'nn_k_y': nn_k_y,
            'nn_nu': nn_nu,
            'nn_velocity': nn_velocity,
            'physics_k_x': phys_k_x,
            'physics_k_y': phys_k_y,
            'physics_velocity': phys_velocity,
            'physics_nu': phys_nu,
            # K-point errors (NN prediction vs ADAM-converged point)
            'k_x_error': k_x_error,
            'k_y_error': k_y_error,
            'k_magnitude_error': k_magnitude_error,
            # Velocity errors (multiple metrics)
            'velocity_error': velocity_error,
            'velocity_relative_error_percent': velocity_relative_error * 100,
            'log_velocity_error': log_velocity_error,
            'nu_error': nu_error,
            'nu_relative_error_percent': nu_relative_error * 100,
            # NN point quality metrics (from benchmark)
            'nn_point_loss': nn_point_loss,
            'nn_point_gap': nn_point_gap,
            'nn_point_r2': nn_point_r2,
            'nn_point_isotropy': nn_point_isotropy,
            # Physics loss (from local computation - DEPRECATED, may remove)
            'nn_physics_loss': nn_physics_loss,
            'physics_physics_loss': physics_physics_loss,
            'loss_ratio': loss_ratio,
            'loss_absolute_diff': loss_absolute_diff
        }

        comparison_details.append(detail_entry)

    def compute_binned_error_analysis(self, comparison_details: List[Dict[str, Any]]) -> Dict[str, Any]:
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
