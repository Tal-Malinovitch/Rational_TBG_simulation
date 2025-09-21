"""
Training data generation for TBG Dirac point prediction neural networks.

This module implements ADAM optimization to find Dirac points in twisted bilayer graphene
systems and generates training datasets for machine learning models. It includes:
- ADAM optimizer implementation for gradient descent
- Band intersection point and Dirac points finding algorithms
- Dirac point optimization with multiple quality metrics
- Training data generation with scale-invariant augmentation
- CSV export functionality
- Time measuring 

The training data generation systematically explores TBG parameter space (twist angles,
interlayer coupling, weights) to create a comprehensive dataset of Dirac point locations.
"""

import constants
from constants import np, logging, dataclasses, gc, time, csv, os, defaultdict
from constants import List, Tuple, Optional, Dict, DefaultDict, Union, Any, Callable
from TBG import tbg, periodic_graph, Dirac_analysis
from utils import compute_twist_constants
from stats import statistics
import json
import multiprocessing as mp
# Module constants
unit_cell_radius_factor_default: float = constants.DEFAULT_UNIT_CELL_RADIUS_FACTOR
DEFAULT_WEIGHTS: List[float] = constants.DEFAULT_NN_LOSS_WEIGHTS  # [gap, R2, isotropy] loss weights
DEFAULT_PARAMS = dataclasses.asdict(constants.simulation_parameters())
DEFAULT_PARAMS_ADAM = dataclasses.asdict(constants.simulation_parameters_adam())

class gradient_decent_adam():
    """
    ADAM (Adaptive Moment Estimation) optimizer for gradient descent.
    
    Combines momentum and adaptive learning rates for efficient optimization.
    Maintains exponential moving averages of gradients and squared gradients.
    
    Attributes:
        alpha (float): Learning rate (step size).
        beta_1 (float): Exponential decay rate for first moment estimates.
        beta_2 (float): Exponential decay rate for second moment estimates.
        eps (float): Small constant to prevent division by zero.
        current_m (float): Current first moment estimate (momentum).
        current_v (float): Current second moment estimate (variance).
        weight (float): Current weight value being optimized.
        t (int): Time step counter for bias correction.
    """
    def __init__(self, alpha: float = DEFAULT_PARAMS_ADAM["alpha"],
                  beta_1: float = DEFAULT_PARAMS_ADAM["beta_1"], 
                 beta_2: float = DEFAULT_PARAMS_ADAM["beta_2"],
                   eps: float = DEFAULT_PARAMS_ADAM["eps"], 
                 initial_weight: float = DEFAULT_PARAMS_ADAM["initial_weight"]) -> None:
        """
        Initialize ADAM optimizer with specified hyperparameters.
        
        Args:
            alpha (float, optional): Learning rate. 
            beta_1 (float, optional): Exponential decay rate for first moment. 
            beta_2 (float, optional): Exponential decay rate for second moment. 
            eps (float, optional): Small constant for numerical stability. 
            initial_weight (float, optional): Initial weight value.
        """
        self.alpha: float = alpha
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.eps: float = eps
        self.current_m: float = 0  # First moment estimate
        self.current_v: float = 0  # Second moment estimate
        self.weight: float = initial_weight
        self.t: int = 0  # Time step counter

    def update(self, gradient: float = 0) -> None:
        """
        Update the weight using ADAM optimization algorithm with gradient clipping.

        Performs gradient clipping, bias-corrected first and second moment estimation,
        then updates the weight using adaptive learning rate.

        Args:
            gradient (float, optional): Current gradient value. Defaults to 0.
        """
        self.t+=1

        # Apply gradient clipping to prevent gradient explosion
        clipped_gradient = np.clip(gradient, -constants.GRADIENT_CLIP_VALUE, constants.GRADIENT_CLIP_VALUE)

        # Update biased first moment estimate
        self.current_m=self.beta_1*self.current_m+(1-self.beta_1)*clipped_gradient
        corrected_m=self.current_m/(1-self.beta_1**self.t)
        # Update biased second moment estimate
        self.current_v=self.beta_2*self.current_v+(1-self.beta_2)*clipped_gradient**2
        corrected_v=self.current_v/(1-self.beta_2**self.t)

        # Update weight using ADAM formula
        self.weight-=corrected_m*self.alpha/(self.eps+np.sqrt(corrected_v))

    def cleanup(self) -> None:
        """Clean up ADAM optimizer by resetting all parameters."""
        self.current_m = 0.0
        self.current_v = 0.0
        self.weight = 0.0
        self.t = 0

    def __enter__(self) -> 'gradient_decent_adam':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

def _adam_optimize_k_point(initial_k: Tuple[float, float],periodic_graph: periodic_graph,
                            compute_metrics_fn: Callable[[List[float]], Tuple[float, float, float, float]],
                            convergence_threshold: float, max_iterations: int, 
                            learning_rate: float = constants.ADAM_LEARNING_RATE,weights: List[float]=None) -> Tuple[Tuple[float, float], float]:
    """
    Generic ADAM optimization for k-point finding.
    
    Args:
        initial_k (Tuple[float, float]): Starting k-point coordinates [k_1, k_2]
        compute_metrics_fn (Callable): Function that computes (metric, grad_k_1, grad_k_2, target_value)
            Takes k_point as List[float] and returns the metric to minimize, gradients, and target value
        convergence_threshold (float): Stopping criterion - optimization stops when metric < threshold  
        max_iterations (int): Maximum number of iterations before giving up
        learning_rate (float, optional): ADAM learning rate. Defaults to constants.ADAM_LEARNING_RATE.
        
    Returns:
        Tuple[Tuple[float, float], float]: Tuple of (optimized_k_point, final_target_value)
        
    Raises:
        ValueError: If optimization fails to converge within max_iterations
    """
    k_1, k_2 = initial_k
    adam_k_1 = gradient_decent_adam(alpha=learning_rate, initial_weight=k_1)
    adam_k_2 = gradient_decent_adam(alpha=learning_rate, initial_weight=k_2)
    prev_metric = 0
    
    for iteration in range(max_iterations):
        if weights is None:
            metrics, grad_k_1, grad_k_2, target_value = compute_metrics_fn((k_1, k_2),periodic_graph)
        else:
            metrics, grad_k_1, grad_k_2, target_value = compute_metrics_fn((k_1, k_2),periodic_graph,weights)
        
        # Check convergence
        if metrics < convergence_threshold:
            k_1 = (k_1 + 0.5) % 1.0 - 0.5
            k_2 = (k_2 + 0.5) % 1.0 - 0.5  
            logging.debug(f"Converged at Iter {iteration:2d}: metric={metrics:.6f} "
                         f"k=({k_1:.4f},{k_2:.4f})")
            return (k_1, k_2), target_value
            
        # Adaptive learning rate - reduce if significant improvement
        if iteration > 0 and metrics < prev_metric * constants.IMPROVEMENT_THRESHOLD:
            adam_k_1.alpha *= constants.LEARNING_RATE_REDUCTION_FACTOR
            adam_k_2.alpha *= constants.LEARNING_RATE_REDUCTION_FACTOR
            # Reset momentum
            adam_k_1.current_m = adam_k_1.current_v = 0
            adam_k_2.current_m = adam_k_2.current_v = 0
            
        # Update positions using ADAM
        adam_k_1.update(grad_k_1)
        adam_k_2.update(grad_k_2)
        
        k_1 += adam_k_1.weight
        k_2 += adam_k_2.weight
        
        # Wrap coordinates to unit cell
        k_1 = (k_1 + 0.5) % 1.0 - 0.5
        k_2 = (k_2 + 0.5) % 1.0 - 0.5
        
        # Reset ADAM weights for next iteration
        adam_k_1.weight = adam_k_2.weight = 0
        prev_metric = metrics
        
        logging.debug(f"Iter {iteration:2d}: metric={metrics:.6f} grad_norm={np.linalg.norm([grad_k_1, grad_k_2]):.4f} "
                     f"k=({k_1:.4f},{k_2:.4f}) lr=({adam_k_1.alpha:.1e},{adam_k_2.alpha:.1e})")
        
    raise ValueError(f"Failed to converge after {max_iterations} iterations, final metric: {metrics:.6f}")
def compute_gap_metrics(k_point: List[float],periodic_graph: periodic_graph) -> Tuple[float, float, float, float]:
    """
    Compute band gap and gradients for intersection point finding.
    
    Args:
        k_point (List[float]): K-point coordinates [k_1, k_2]
        periodic_graph (periodic_graph): Periodic graph object for TBG system.
        
    Returns:
        Tuple[float, float, float, float]: (gap, grad_k_1, grad_k_2, gap_return_value)
    """
    # Ensure adjacency matrix is built
    if periodic_graph.adj_matrix is None or periodic_graph.periodic_edges is None:
        periodic_graph.build_adj_matrix(DEFAULT_PARAMS['inter_graph_weight'], DEFAULT_PARAMS['intra_graph_weight'])
        
    # Compute eigenvalues and derivatives
    eigvals, der_x_gap, der_y_gap = periodic_graph.compute_energy_and_derivative_of_energy_at_a_point(k_point, 1)
    
    gap = abs(eigvals[1] - eigvals[0])
    
    # Determine sign for gradient direction
    sign_of_gap = 1 if eigvals[1] > eigvals[0] else -1
    
    der_x = sign_of_gap * np.real(der_x_gap)  # Ensure real values
    der_y = sign_of_gap * np.real(der_y_gap)
    
    # Transform gradients from Cartesian to lattice coordinates
    B = np.array([periodic_graph.dual_vectors[0], periodic_graph.dual_vectors[1]]).T
    B_inv = np.linalg.inv(B)
    
    grad_cartesian = np.array([der_x, der_y])
    grad_lattice = B_inv.T @ grad_cartesian
    
    return gap, grad_lattice[0], grad_lattice[1], gap
def find_intersection_point(initial_k: List[float], periodic_graph: periodic_graph,
                             max_num_of_iterations: int = constants.MAX_ITERATIONS) -> Tuple[Tuple[float, float], float]:
    """
    Find intersection point between the bands using ADAM optimization.
    
    Args:
        initial_k (List[float]): Initial guess [k_1, k_2] in relative coordinates.
        periodic_graph (periodic_graph): Periodic graph object for TBG system.
        max_num_of_iterations (int, optional): Maximum number of optimization iterations. 
            Defaults to constants.MAX_ITERATIONS.
        
    Returns:
        Tuple[Tuple[float, float], float]: Tuple containing:
            - (k_1, k_2): Coordinates of intersection point in relative coordinates.
            - gap: Energy gap at the intersection point.
        
    Raises:
        ValueError: If optimization fails to converge after max_num_of_iterations.
    """
    
    return _adam_optimize_k_point(tuple(initial_k),periodic_graph, compute_gap_metrics, 
        constants.DEFAULT_E_TOLERANCE, max_num_of_iterations,constants.ADAM_LEARNING_RATE)
def compute_loss_metrics(k_point: List[float],
                         periodic_graph: periodic_graph,weights: List[float],) -> Tuple[float, float, float, float]:
        """
        Compute loss function and gradients for Dirac point finding.
        
        Args:
            k_point (List[float]): K-point coordinates [k_1, k_2]
            periodic_graph (periodic_graph): Periodic graph object for TBG system.
            
        Returns:
            Tuple[float, float, float, float]: (loss, grad_k_1, grad_k_2, dirac_velocity)
        """
        dirac_analyzer = Dirac_analysis(periodic_graph)
        metrics, metrics_der_k_1, metrics_der_k_2, mu_v, _ = dirac_analyzer.check_Dirac_point(k_point, 1)
        
        loss = np.sum(np.array(metrics) * np.array(weights))
        grad_k_1 = np.sum(np.array(metrics_der_k_1) * np.array(weights))
        grad_k_2 = np.sum(np.array(metrics_der_k_2) * np.array(weights))
        
        return loss, grad_k_1, grad_k_2, mu_v
def Find_Dirac_point(initial_k: List[float], periodic_graph: periodic_graph, 
                     weights: List[float], max_num_of_iterations: int = constants.MAX_ITERATIONS) -> Tuple[Tuple[float, float], float]:
    """
    Find Dirac point using ADAM optimization.
    
    Args:
        initial_k (List[float]): Initial guess [k_1, k_2] in relative coordinates.
        periodic_graph (periodic_graph): Periodic graph object for TBG system.
        weights (List[float]): Loss function weights [gap, R2, isotropy].
        max_num_of_iterations (int, optional): Maximum number of optimization iterations. 
            Defaults to constants.MAX_ITERATIONS.
        
    Returns:
        Tuple[Tuple[float, float], float]: Tuple containing:
            - (k_1, k_2): Coordinates of Dirac point in relative coordinates.
            - Dirac velocity at the found point.
        
    Raises:
        ValueError: If optimization fails to converge after max_num_of_iterations.
    """

        
    return _adam_optimize_k_point(tuple(initial_k), periodic_graph, compute_loss_metrics,
        constants.DEFAULT_TOLERANCE, max_num_of_iterations, constants.ADAM_LEARNING_RATE, weights)
def find_Dirac_point_grid(center: List[float], periodic_graph: periodic_graph, 
                     weights: List[float], radius: float = constants.DEFAULT_SEARCH_RADIUS,
                     num_of_grid_points: int = constants.DEFAULT_NUM_OF_POINTS,
                     N_scale: float = 0.0) -> List[Tuple[Tuple[float, float], float]]:
    """
    First it runs a grid search to find all the intersection points,
    Then it uses these intersection point as a starting point for trying to get the exact Dirac point. 

    
    Args:
        center: The center of the search grid [k_1, k_2] - in relative coordinates
        periodic_graph: Periodic graph object
        weights: Loss function weights [gap, R2, isotropy]
        radius: the radius of the search grid.
        num_of_grid_points: the number of grid points.
        N_scale : the scaling factor 
    Returns:
        List of tuples of (k_1, k_2) coordinates of Dirac points- in relative coordinates
        
    Raises:
        ValueError: If N_factor is not provided.
        ValueError: If optimization fails to converge

    """
    results=[]
    if N_scale==0:
        raise ValueError("You must provide the N scale of the system!")
    max_num_of_iteration=constants.MAX_ITERATIONS
    #Adapt the number of grid points to the scale of N.
    # Since as N gets bigger- the Brioulin zone is getting smaller, and the system becomes bigger
    #also adapt the maximal number of iteration - to preserve the time scale
    if N_scale>10:
        num_of_grid_points-=2
        max_num_of_iteration/=3
        max_num_of_iteration=int(max_num_of_iteration)
    elif N_scale>5:
        num_of_grid_points-=1
        max_num_of_iteration/=2
        max_num_of_iteration=int(max_num_of_iteration)
    for i in range(num_of_grid_points):
        for j in range(num_of_grid_points):
            offset_x = (i - num_of_grid_points//2) * radius / num_of_grid_points
            offset_y = (j - num_of_grid_points//2) * radius / num_of_grid_points
            
            test_k = [center[0] + offset_x, center[1] + offset_y]
            
            try:
                result,gap = find_intersection_point(test_k, periodic_graph,max_num_of_iteration)
                results.append({'k': (result[0], result[1]),'gap': gap})
            except ValueError:
                logging.debug(f"Didn't find intersection point from k=({test_k[0]:.2f},{test_k[1]:.2f})!")
                continue
    if len(results)==0:
        logging.info(f"Didn't find intersection points!")
        raise ValueError(f"Failed to find intersection points!")
    results_unique=[]
    for result in results:
        is_duplicate=False
        for unique_result in results_unique:
            k_diff=np.linalg.norm([result['k'][0]-unique_result['k'][0],result['k'][1]-unique_result['k'][1]])
            k_diff_2=np.linalg.norm([result['k'][0]+unique_result['k'][0],result['k'][1]+unique_result['k'][1]]) # We remove the ppoint induced by symmetry of k->-k.
            if k_diff<constants.DEFAULT_K_TOLERANCE or k_diff_2<constants.DEFAULT_K_TOLERANCE:
                if result['gap'] < unique_result['gap']:
                    unique_result['k'] = result['k']
                    unique_result['gap'] = result['gap']
                    dirac_analyzer = Dirac_analysis(periodic_graph)
                    metrics, _, _,mu_v,_ = dirac_analyzer.check_Dirac_point(( result['k'][0],result['k'][1]), 1)
                    loss = np.sum(np.array(metrics) * np.array(weights))
                    unique_result['loss'] = loss
                    unique_result['mu_v'] = mu_v
                is_duplicate=True
                break
        if not is_duplicate:
            dirac_analyzer = Dirac_analysis(periodic_graph)
            metrics, _, _ ,mu_v,_= dirac_analyzer.check_Dirac_point(( result['k'][0],result['k'][1]), 1)
            loss = np.sum(np.array(metrics) * np.array(weights))
            new_result = {
                'k': result['k'],
                'gap': result['gap'],
                'loss': loss,
                'mu_v': mu_v
            }
            results_unique.append(new_result)

    logging.info(f"Found {len(results_unique)} unique intersection points from {len(results)} total")
    all_dirac_points = []
    for  result in results_unique:
        try:
            logging.info(f"Starting at k=({result['k'][0]:.4f}, {result['k'][1]:.4f}) with loss={result['loss']:.6f}")
            result_dirac = Find_Dirac_point((result['k'][0],result['k'][1]), periodic_graph, weights,max_num_of_iteration)
            if result_dirac is not None:
                all_dirac_points.append(result_dirac)
        except ValueError:
            logging.debug(f"Failed at finding a Dirac point from k=({result['k'][0]:.4f}, {result['k'][1]:.4f}) with loss={result['loss']:.6f}")
            continue
    logging.info(f"Found {len(all_dirac_points)} Dirac points from {len(results_unique)} total")
    if len(all_dirac_points)==0:
        raise ValueError("Failed to extract Dirac points from intersections!")

    return all_dirac_points
def compute_sym_factor(a: int, b: int) -> List[int]:
    """
    Computes the symmetric factors for twist angle parameters.

    Args:
        a (int): First twist angle parameter.
        b (int): Second twist angle parameter.
        
    Returns:
        List[int]: List of the new a and b according to:
            a_new = (a + 3*b) / 2^(eps + eps'),
            b_new = (a - b) / 2^(eps + eps'),
            where eps = 1 if (a*b) % 2 = 1, and 0 otherwise,
            and eps' = 1 if (a*b) % 4 = 1, and 0 otherwise.
    """
    factor=1
    if (a*b)%2==1:
        factor*=2
    if (a*b)%4==1:
        factor*=2
    a_new=(a+3*b)/factor
    a_new=int(a_new)
    b_new=(a-b)/factor
    b_new=int(b_new)
    return([a_new,b_new])

def _create_training_samples(a: int, b: int,interlayer_dist_threshold: float,
    intralayer_dist_threshold: float,inter_graph_weight: float,intra_graph_weight: float,
    N_scale: float,num_nodes: int,dirac_point: Tuple[Tuple[float, float], float]) -> List[List[Union[int, float]]]:
    """
    Create multiple training samples with scaled parameters for data augmentation.
    
    Args:
        a (int): Integer twist parameter a
        b (int): Integer twist parameter b  
        interlayer_dist_threshold (float): Interlayer coupling threshold
        intralayer_dist_threshold (float): Intralayer coupling threshold
        inter_graph_weight (float): Inter-sublattice weight
        intra_graph_weight (float): Intra-sublattice weight
        N_scale (float): Scaling factor for the system
        num_nodes (int): Number of nodes in the graph
        dirac_point (Tuple[Tuple[float, float], float]): ((k_x, k_y), dirac_velocity)
        
    Returns:
        List[List[Union[int, float]]]: List of training samples, each containing
            [a_scaled, b_scaled, interlayer_threshold, intralayer_threshold, 
             inter_weight, intra_weight, N_scale, num_nodes, k_x, k_y, dirac_velocity]
    """
    samples = []
    for factor in range(constants.TRAINING_SAMPLE_MIN_FACTOR, constants.TRAINING_SAMPLE_MAX_FACTOR):
        sample = [
            factor * a,  # Scaled a parameter
            factor * b,  # Scaled b parameter  
            interlayer_dist_threshold,
            intralayer_dist_threshold,
            inter_graph_weight,
            intra_graph_weight,
            N_scale,
            num_nodes,
            dirac_point[0][0],  # k_x
            dirac_point[0][1],  # k_y  
            dirac_point[1]      # dirac_velocity
        ]
        samples.append(sample)
    return samples

def _process_tbg_system_worker(system_params: Tuple[int, int, float, float, float, List[float]]) -> Tuple[List[List[Union[int, float]]], List, List, Dict, List]:
    """
    Worker function to process one TBG system through all weight combinations.
    
    This function processes a single (a, b, intralayer_dist_threshold, interlayer_dist_threshold) 
    combination through all weight variations, preserving the matrix reuse optimization.
    
    Args:
        system_params: Tuple containing (a, b, intralayer_dist_threshold, interlayer_dist_threshold, N_scale, weights)
        
    Returns:
        Tuple containing (training_samples, failed_cases, no_intersection_cases, system_stats, timing_data)
    """
    a, b, intralayer_dist_threshold, interlayer_dist_threshold, N_scale, weights = system_params
    
    # Initialize worker-local variables
    training_batch = []
    failed_cases = []
    no_intersection_cases = []
    timing_data = []  # Store timing data for each combination
    system_stats = {
        'total_combinations': 0,
        'successful_combinations': 0,
        'no_intersection_combinations': 0,
        'total_dirac_points': 0
    }
    
    try:
        # Compute TBG parameters (same as original)
        _, _, factor, k_point = compute_twist_constants(a, b)
        n = int(constants.np.round(factor * N_scale * unit_cell_radius_factor_default))
        
        # Build TBG structure using context manager
        with tbg(n, n, a, b, interlayer_dist_threshold, intralayer_dist_threshold,
                unit_cell_radius_factor_default) as TBG_graph:
            with TBG_graph.full_graph.create_periodic_copy(TBG_graph.lattice_vectors, k_point) as periodic_graph:
                
                # Process all weight combinations for this TBG system
                build_adj_matrix_flag = True
                # More strategic weight sampling - 3 key values for better coverage
                intra_weights = [0.5, 1.0, 1.5]
                
                for intra_graph_weight in intra_weights:
                    # Fixed ratios for better predictable coverage
                    inter_weight_ratios = [0.3, 0.5, 0.7, 0.9]  # Relative to intra_weight
                    inter_weights = [ratio * intra_graph_weight for ratio in inter_weight_ratios]
                    
                    for inter_graph_weight in inter_weights:
                        combination_start = time.time()
                        success = False
                        no_intersection = False
                        num_of_Dirac = 0
                        
                        try:
                            if build_adj_matrix_flag:
                                # Build adjacency matrix (same as original)
                                periodic_graph.build_adj_matrix(inter_graph_weight, intra_graph_weight)
                                build_adj_matrix_flag = False
                            else:
                                # Just update weights (preserving optimization)
                                periodic_graph.update_weights(inter_graph_weight, intra_graph_weight)
                            
                            # Find Dirac points (same as original)
                            Dirac_points = find_Dirac_point_grid(k_point, periodic_graph, weights,
                                                                constants.DEFAULT_SEARCH_RADIUS,
                                                                constants.DEFAULT_NUM_OF_POINTS, N_scale)
                            
                            if not Dirac_points:
                                logging.warning(f"No Dirac points found for (a={a}, b={b})")
                                continue
                                
                            num_of_Dirac = len(Dirac_points)
                            
                            # Create training samples (same as original)
                            for Dirac_point in Dirac_points:
                                # Compute symmetric case
                                sym_case = compute_sym_factor(a, b)
                                
                                # Add samples for symmetric case
                                sym_samples = _create_training_samples(
                                    sym_case[0], sym_case[1],
                                    interlayer_dist_threshold, intralayer_dist_threshold,
                                    inter_graph_weight, intra_graph_weight,
                                    N_scale, len(TBG_graph.full_graph.nodes),
                                    Dirac_point
                                )
                                training_batch.extend(sym_samples)
                                
                                # Add samples for original case
                                original_samples = _create_training_samples(
                                    a, b,
                                    interlayer_dist_threshold, intralayer_dist_threshold,
                                    inter_graph_weight, intra_graph_weight,
                                    N_scale, len(TBG_graph.full_graph.nodes),
                                    Dirac_point
                                )
                                training_batch.extend(original_samples)
                            
                            success = True
                            system_stats['successful_combinations'] += 1
                            system_stats['total_dirac_points'] += num_of_Dirac
                            
                            logging.info(f"Worker success: (a={a}, b={b}, inter_thresh={interlayer_dist_threshold:.2f}, "
                                        f"intra_thresh={intralayer_dist_threshold:.2f}, intra={intra_graph_weight:.1f}, "
                                        f"inter={inter_graph_weight:.1f}) -> Found {len(Dirac_points)} Dirac points")
                            
                        except (ValueError, Exception) as e:
                            failed_case = [a, b, interlayer_dist_threshold, intralayer_dist_threshold, 
                                          intra_graph_weight, inter_graph_weight]
                            
                            error_msg = str(e)
                            if "Failed to find intersection points!" in error_msg:
                                no_intersection_cases.append(failed_case)
                                no_intersection = True
                                system_stats['no_intersection_combinations'] += 1
                            else:
                                failed_cases.append(failed_case)
                                
                            logging.warning(f"Worker failed case: (a={a}, b={b}, "
                                           f"inter_thresh={interlayer_dist_threshold:.1f}, "
                                           f"intra_thresh={intralayer_dist_threshold:.1f}, "
                                           f"intra={intra_graph_weight:.1f}, "
                                           f"inter={inter_graph_weight:.1f}): {str(e)}")
                        
                        # Record timing data for this combination
                        combination_duration = time.time() - combination_start
                        timing_record = {
                            'duration': combination_duration,
                            'system_size': len(TBG_graph.full_graph.nodes),
                            'n_scale': N_scale,
                            'success': success,
                            'no_intersection': no_intersection,
                            'num_of_Dirac': num_of_Dirac
                        }
                        timing_data.append(timing_record)
                        
                        system_stats['total_combinations'] += 1
                
                # Automatic cleanup via context managers
                gc.collect()
        
        logging.info(f"Worker completed: (a={a}, b={b}) -> {len(training_batch)} samples, "
                    f"{system_stats['successful_combinations']}/{system_stats['total_combinations']} successful")
        
    except Exception as e:
        logging.error(f"Worker failed for TBG system (a={a}, b={b}): {str(e)}")
        
    return training_batch, failed_cases, no_intersection_cases, system_stats, timing_data


def create_training_data(batch_size: int = constants.DEFAULT_BATCH_SIZE, resume_from_checkpoint: bool = True, 
                        use_parallel: bool = True, num_processes: Optional[int] = None) -> int:
    """
    Generate training data for Dirac point neural network with batch processing.
    
    This function systematically explores the TBG parameter space by varying twist angle
    parameters (a, b), interlayer and intralayer distance thresholds, and graph weights.
    For each parameter combination, it attempts to find Dirac points and creates multiple
    training samples with scale-invariant augmentation.
    
    Args:
        batch_size (int): Number of training samples to accumulate before writing to disk
        resume_from_checkpoint (bool): Whether to resume from previous checkpoint
        use_parallel (bool): Whether to use multiprocessing parallelization
        num_processes (Optional[int]): Number of processes to use. If None, uses cpu_count()
    
    Returns:
        int: Total number of training samples generated
            
    Raises:
        ValueError: If critical errors occur during parameter exploration that prevent
                   training data generation.
    """
    # Initialize everything:
    weights = DEFAULT_WEIGHTS
    training_batch = []
    total_samples_generated = 0
    failed_cases = []
    no_intersection_cases = []
    cases_covered = []
    # Selective a values for strategic magic angle coverage
    # This targets specific high-value angles while minimizing computational cost
    # Gets angles down to 1.741° (very close to magic angle ~1.107°)
    a_values = [2, 3, 4, 5, 6, 7, 13, 15]  
    intralayer_thresholds = np.array([1.0,1.2,1.5,1.8])
    
    # Initialize the statistics:
    stats = statistics()
    stats.start_time = time.time()
    
    # Load checkpoint if resuming
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = load_checkpoint()
        if checkpoint:
            cases_covered = checkpoint.get('cases_covered', [])
            stats.total_combinations = checkpoint.get('total_combinations', 0)
            total_samples_generated = checkpoint.get('total_samples_generated', 0)
            logging.info(f"Resuming from checkpoint: {stats.total_combinations} combinations processed, {total_samples_generated} samples generated")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    logging.info(f"Using {'parallel' if use_parallel else 'sequential'} processing with {num_processes if use_parallel else 1} {'processes' if use_parallel else 'process'}")
    
    if use_parallel:
        # Parallel processing path
        return _create_training_data_parallel(weights, cases_covered, stats, total_samples_generated, 
                                            a_values, intralayer_thresholds, batch_size, num_processes)
    else:
        # Sequential processing path (original implementation)
        return _create_training_data_sequential(weights, cases_covered, stats, total_samples_generated, 
                                               a_values, intralayer_thresholds, batch_size)


def _create_training_data_parallel(weights: List[float], cases_covered: List, stats, total_samples_generated: int,
                                 a_values: range, intralayer_thresholds: np.ndarray, 
                                 batch_size: int, num_processes: int) -> int:
    """
    Parallel implementation of training data generation.
    """
    # Build list of TBG system parameters to process in parallel
    system_params_list = []
    
    for a in a_values:
        # Avoid the trivial symmetry case
        if a == 3:
            continue
            
        for b in range(1, a):
            if np.gcd(a, b) != 1:  # Skip non-coprime pairs
                continue
            if [a, b] in cases_covered:  # Skip already covered cases
                continue
                
            # Compute TBG constants once per (a,b) pair
            N_scale, _, factor, k_point = compute_twist_constants(a, b)
            
            # Mark symmetric cases as covered
            sym_case = compute_sym_factor(a, b)
            cases_covered.append([a, b])
            cases_covered.append(sym_case)
            
            for intralayer_dist_threshold in intralayer_thresholds:
                # Use fixed ratios for better predictable coverage
                interlayer_ratios = [0.3, 0.5, 0.7, 0.9]  # Relative to intralayer
                interlayer_thresholds = [ratio * intralayer_dist_threshold for ratio in interlayer_ratios]
                
                for interlayer_dist_threshold in interlayer_thresholds:
                    # Each system_params entry represents one TBG system to be processed
                    system_params = (a, b, intralayer_dist_threshold, interlayer_dist_threshold, N_scale, weights)
                    system_params_list.append(system_params)
    
    logging.info(f"Processing {len(system_params_list)} TBG systems in parallel using {num_processes} processes")
    
    # Process systems in parallel
    training_batch = []
    all_failed_cases = []
    all_no_intersection_cases = []
    
    with mp.Pool(processes=num_processes) as pool:
        # Process systems in parallel and get results as they complete
        results_iterator = pool.imap_unordered(_process_tbg_system_worker, system_params_list)
        
        # Process results as they arrive (not waiting for all to complete)
        for worker_training_batch, worker_failed_cases, worker_no_intersection_cases, worker_stats, worker_timing_data in results_iterator:
            training_batch.extend(worker_training_batch)
            all_failed_cases.extend(worker_failed_cases)
            all_no_intersection_cases.extend(worker_no_intersection_cases)
            
            # Update global statistics (only counts, timing data handled separately)
            stats.total_combinations += worker_stats['total_combinations']
            stats.successful_combinations += worker_stats['successful_combinations']
            stats.successful_combinations_num_of_Dirac += worker_stats['total_dirac_points']
            stats.failed_no_intersections += worker_stats['no_intersection_combinations']
            stats.failed_no_Dirac += (worker_stats['total_combinations'] - 
                                     worker_stats['successful_combinations'] - 
                                     worker_stats['no_intersection_combinations'])
            
            # Integrate worker timing data into main statistics (timing only, not counts)
            for timing_record in worker_timing_data:
                stats.combination_times.append(timing_record['duration'])
                stats.system_size_times[timing_record['system_size']].append(timing_record['duration'])
                stats.n_scale_times[round(timing_record['n_scale'], 1)].append(timing_record['duration'])
            
            total_samples_generated += len(worker_training_batch)
            
            # Write batch to disk if it's getting large
            if len(training_batch) >= batch_size:
                append_training_data_batch(training_batch)
                
                # Save statistics periodically (for crash recovery)
                stats.log_statistics()
                stats.save_statistics()
                
                # Save checkpoint with current progress
                checkpoint_data = {
                    'total_combinations': stats.total_combinations,
                    'total_samples_generated': total_samples_generated,
                    'cases_covered': cases_covered,
                    'timestamp': time.time()
                }
                save_checkpoint(checkpoint_data)
                
                training_batch.clear()
                gc.collect()
    
    # Write any remaining samples
    if training_batch:
        append_training_data_batch(training_batch)
        total_samples_generated += len(training_batch)
    
    # Final statistics
    stats.log_statistics()
    stats.save_statistics()
    
    logging.info(f"Parallel training completed: {total_samples_generated} samples generated")
    logging.info(f"Failed cases: {len(all_failed_cases)}")
    logging.info(f"No intersection cases: {len(all_no_intersection_cases)}")
    
    return total_samples_generated


def _create_training_data_sequential(weights: List[float], cases_covered: List, stats, total_samples_generated: int,
                                   a_values: range, intralayer_thresholds: np.ndarray, batch_size: int) -> int:
    """
    Sequential implementation of training data generation (original algorithm).
    """
    # This would contain the original nested loop implementation
    # For brevity, I'll reference the original code that was replaced
    logging.info("Using sequential processing (original algorithm)")
    
    # Insert original nested loop code here if sequential mode needed
    # For now, raise NotImplementedError to force parallel usage
    raise NotImplementedError("Sequential mode not fully implemented in this refactor. Use use_parallel=True")


def save_training_data(training_data: List[List[Union[int, float]]], filename: str = constants.PATH + "/dirac_training_data.csv") -> None:
    """
    Save training data to CSV file with appropriate headers.
    
    Args:
        training_data (List[List[Union[int, float]]]): List of training samples, each containing:
            [a, b, interlayer_dist_threshold, intralayer_dist_threshold, 
             inter_graph_weight, intra_graph_weight, N_scale, num_nodes, 
             target_k_x, target_k_y, dirac_velocity]
        filename (str, optional): Output CSV file path. Defaults to constants.PATH + "/dirac_training_data.csv".
    """

    
    headers = [
        'a', 'b', 'interlayer_dist_threshold', 'intralayer_dist_threshold', 'inter_graph_weight', 'intra_graph_weight',
        'N_scale', 'num_nodes', 'target_k_x', 'target_k_y',"Dirac_velocity"
    ]
    
    # Ensure directory exists before creating file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(training_data)
    
    logging.info(f"Training data saved to {filename} ({len(training_data)} samples)")

def append_training_data_batch(training_batch: List[List[Union[int, float]]], filename: str = constants.PATH + "/dirac_training_data.csv") -> None:
    """
    Append a batch of training data to CSV file. Creates file with headers if it doesn't exist.
    
    Args:
        training_batch (List[List[Union[int, float]]]): Batch of training samples
        filename (str): CSV file path
    """

    
    headers = [
        'a', 'b', 'interlayer_dist_threshold', 'intralayer_dist_threshold', 'inter_graph_weight', 'intra_graph_weight',
        'N_scale', 'num_nodes', 'target_k_x', 'target_k_y', "Dirac_velocity"
    ]
    
    # Ensure directory exists before creating file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(training_batch)
    
    logging.info(f"Appended {len(training_batch)} training samples to {filename}")

def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_file: str = constants.PATH + "/training_checkpoint.json") -> None:
    """
    Save checkpoint data for crash recovery.
    
    Args:
        checkpoint_data (Dict): Dictionary containing current progress state
        checkpoint_file (str): Checkpoint file path
    """

    
    try:
        # Ensure directory exists before creating file
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logging.info(f"Checkpoint saved to {checkpoint_file}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file: str = constants.PATH + "/training_checkpoint.json") -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data for crash recovery.
    
    Args:
        checkpoint_file (str): Checkpoint file path
        
    Returns:
        Dict or None: Checkpoint data if found, None otherwise
    """

    
    if not os.path.exists(checkpoint_file):
        return None
        
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        logging.info(f"Checkpoint loaded from {checkpoint_file}")
        return checkpoint_data
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # You can adjust batch_size based on your memory constraints
    # Smaller batch_size = more frequent disk writes, less memory usage
    # Larger batch_size = fewer disk writes, more memory usage
    batch_size = constants.DEFAULT_BATCH_SIZE  # Adjust as needed
    
    # Use parallel processing by default (set use_parallel=False to use sequential)
    total_samples = create_training_data(batch_size=batch_size, resume_from_checkpoint=True, 
                                       use_parallel=True, num_processes=None)
    
    if total_samples > 0:
        logging.info(f"Training data generation completed successfully with {total_samples} samples!")
    else:
        logging.error("No training data generated!")

