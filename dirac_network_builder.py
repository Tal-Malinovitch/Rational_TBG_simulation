"""
Neural network architecture and TBG graph initialization for Dirac point prediction.

This module handles the specific neural network construction for TBG Dirac point
finding, including mixed activation functions, TBG graph initialization, and
parameter validation following project coding standards.

Classes:
    dirac_network_builder: Constructs and initializes Dirac point prediction networks
"""

import constants
from constants import np, logging
from constants import List, Tuple, Union, Optional
from TBG import tbg, periodic_graph, Dirac_analysis
from utils import compute_twist_constants, validate_ab
from neural_network_base import (
    neural_network, leaky_relu, leaky_relu_der, wrapped_k_activation, 
    wrapped_k_activation_der, dummy_neuron
)

# Configure logging
logger = logging.getLogger(__name__)

# Default network configuration
default_network_config = {
    'input_features': 6,  # a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight
    'output_features': 3,  # k_x, k_y, velocity
    'hidden_layer_size': constants.DEFAULT_NN_LAYER_SIZE,
    'num_hidden_layers': 2,  # Back to 2 layers - additional layer caused worse gradient explosion
    'unit_cell_radius_factor': constants.DEFAULT_UNIT_CELL_RADIUS_FACTOR
}


class dirac_network_builder:
    """
    Constructs neural networks specifically designed for TBG Dirac point prediction.
    
    This class handles the specialized architecture requirements including mixed
    activation functions for k-point wrapping, TBG graph initialization, and
    parameter validation using project utilities.
    
    Attributes:
        network_config (dict): Configuration parameters for network architecture
        current_network (neural_network): Currently built neural network
        current_tbg_params (Optional[dict]): Current TBG system parameters
        current_graph (Optional[tbg]): Current TBG graph structure
        current_periodic_graph (Optional[periodic_graph]): Current periodic graph for physics
    """
    
    def __init__(self, network_config: Optional[dict] = None) -> None:
        """
        Initialize the Dirac network builder with configuration.
        
        Args:
            network_config (dict, optional): Custom network configuration parameters.
                Defaults to default_network_config if not provided.
                
        Raises:
            constants.physics_parameter_error: If network configuration is invalid
        """
        try:
            self.network_config = network_config or default_network_config.copy()
            self._validate_network_config()
            
            # Initialize state
            self.current_network: Optional[neural_network] = None
            self.current_tbg_params: Optional[dict] = None
            self.current_graph: Optional[tbg] = None
            self.current_periodic_graph: Optional[periodic_graph] = None
            
            logger.info(f"dirac_network_builder initialized with config: {self.network_config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dirac_network_builder: {str(e)}")
            raise constants.physics_parameter_error(f"dirac_network_builder initialization failed: {str(e)}")
    
    def _validate_network_config(self) -> None:
        """
        Validate network configuration parameters.
        
        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        required_keys = ['input_features', 'output_features', 'hidden_layer_size', 'num_hidden_layers']
        
        for key in required_keys:
            if key not in self.network_config:
                raise constants.physics_parameter_error(f"Missing required config key: {key}")
        
        # Validate positive integer values
        for key in required_keys:
            value = self.network_config[key]
            if not isinstance(value, int) or value <= 0:
                raise constants.physics_parameter_error(f"Config {key} must be positive integer, got {value}")
        
        # Validate output features is 3 for Dirac point prediction
        if self.network_config['output_features'] != 3:
            raise constants.physics_parameter_error("Output features must be 3 for Dirac point prediction (k_x, k_y, velocity)")
    
    def build_network(self) -> neural_network:
        """
        Build a neural network with Dirac point prediction architecture.
        
        Creates a network with:
        - Input layer for TBG parameters
        - Hidden layers with ReLU activation  
        - Mixed output layer (wrapped k-points + scaled sigmoid velocity)
        
        Returns:
            neural_network: Constructed network ready for training/prediction
            
        Raises:
            constants.physics_parameter_error: If network construction fails
        """
        try:
            # Create dummy inputs for initialization
            dummy_inputs = [0.0] * self.network_config['input_features']
            
            # Define layer structure
            layers_structure = []
            
            # Add hidden layers with Leaky ReLU activation to prevent dead neurons
            for _ in range(self.network_config['num_hidden_layers']):
                layers_structure.append((
                    self.network_config['hidden_layer_size'],
                    leaky_relu,
                    leaky_relu_der
                ))
            
            # Add mixed output layer
            layers_structure.append((
                self.network_config['output_features'],
                'mixed_output',  # Special marker for mixed activation
                'mixed_output'
            ))
            
            # Create the network
            self.current_network = neural_network(
                layers_structure=layers_structure,
                inputs=dummy_inputs,
                loss_function_and_grad=None
            )
            
            logger.info(f"Built neural network with {len(layers_structure)} layers")
            logger.debug(f"Layer structure: {[layer[0] for layer in layers_structure]}")
            
            return self.current_network
            
        except Exception as e:
            logger.error(f"Failed to build neural network: {str(e)}")
            raise constants.physics_parameter_error(f"Network construction failed: {str(e)}")
    
    def validate_tbg_parameters(self, params: List[Union[int, float]]) -> dict:
        """
        Validate and parse TBG parameters using project utilities.
        
        Args:
            params (List[Union[int, float]]): Parameter list containing
                [a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight]
                
        Returns:
            dict: Parsed and validated TBG parameters
            
        Raises:
            constants.physics_parameter_error: If parameters are invalid
        """
        try:
            if len(params) != self.network_config['input_features']:
                raise constants.physics_parameter_error(
                    f"Expected {self.network_config['input_features']} parameters, got {len(params)}"
                )
            
            # Extract and validate basic parameters
            a, b = int(params[0]), int(params[1])
            interlayer_threshold = float(params[2])
            intralayer_threshold = float(params[3])
            inter_weight = float(params[4])
            intra_weight = float(params[5])
            
            # Use project utility for TBG parameter validation
            validate_ab(a, b)
            
            # Validate physical parameters
            constants.validate_positive_number(interlayer_threshold, "interlayer_threshold")
            constants.validate_positive_number(intralayer_threshold, "intralayer_threshold")
            constants.validate_positive_number(inter_weight, "inter_weight")
            constants.validate_positive_number(intra_weight, "intra_weight")
            
            # Compute TBG constants using project utility
            n_scale, alpha, factor, k_point = compute_twist_constants(a, b)
            
            validated_params = {
                'a': a,
                'b': b,
                'interlayer_threshold': interlayer_threshold,
                'intralayer_threshold': intralayer_threshold,
                'inter_weight': inter_weight,
                'intra_weight': intra_weight,
                'n_scale': n_scale,
                'alpha': alpha,
                'factor': factor,
                'k_point': k_point
            }
            
            logger.debug(f"Validated TBG parameters: a={a}, b={b}, n_scale={n_scale:.3f}")
            
            return validated_params
            
        except constants.physics_parameter_error:
            raise
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            raise constants.physics_parameter_error(f"Invalid TBG parameters: {str(e)}")
    
    def initialize_tbg_graphs(self, validated_params: dict) -> Tuple[tbg, periodic_graph]:
        """
        Initialize TBG graph structures for physics-based computations.
        
        Args:
            validated_params (dict): Validated TBG parameters from validate_tbg_parameters()
            
        Returns:
            Tuple[tbg, periodic_graph]: TBG graph and its periodic copy
            
        Raises:
            constants.graph_construction_error: If graph initialization fails
        """
        try:
            # Calculate system size
            unit_cell_radius_factor = self.network_config.get(
                'unit_cell_radius_factor', 
                default_network_config['unit_cell_radius_factor']
            )
            
            n = int(np.round(
                validated_params['factor'] * 
                validated_params['n_scale'] * 
                unit_cell_radius_factor
            ))
            
            # Build TBG graph
            self.current_graph = tbg(
                n, n,
                validated_params['a'], validated_params['b'],
                validated_params['interlayer_threshold'],
                validated_params['intralayer_threshold'],
                unit_cell_radius_factor
            )
            
            # Create periodic copy for physics computations
            self.current_periodic_graph = self.current_graph.full_graph.create_periodic_copy(
                self.current_graph.lattice_vectors,
                validated_params['k_point']
            )
            
            # Build adjacency matrix
            self.current_periodic_graph.build_adj_matrix(
                validated_params['inter_weight'],
                validated_params['intra_weight']
            )
            
            logger.info(f"Initialized TBG graphs: system size n={n}, nodes={len(self.current_graph.full_graph.nodes)}")
            
            return self.current_graph, self.current_periodic_graph
            
        except Exception as e:
            logger.error(f"TBG graph initialization failed: {str(e)}")
            raise constants.graph_construction_error(f"Failed to initialize TBG graphs: {str(e)}")
    
    def set_network_parameters(self, params: List[Union[int, float]]) -> dict:
        """
        Set and validate TBG parameters for the current network.
        
        This method combines parameter validation, graph initialization, and
        network input setting in one convenient call.
        
        Args:
            params (List[Union[int, float]]): TBG parameters
            
        Returns:
            dict: Validated TBG parameters
            
        Raises:
            constants.physics_parameter_error: If no network is built or parameters invalid
            constants.graph_construction_error: If graph initialization fails
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network built. Call build_network() first.")
        
        try:
            # Validate parameters
            validated_params = self.validate_tbg_parameters(params)
            self.current_tbg_params = validated_params
            
            # Initialize TBG graphs
            self.initialize_tbg_graphs(validated_params)
            
            # Set network inputs
            self._set_network_inputs(np.array(params))
            
            logger.debug(f"Set network parameters: {validated_params}")
            
            return validated_params
            
        except (constants.physics_parameter_error, constants.graph_construction_error):
            raise
        except Exception as e:
            logger.error(f"Failed to set network parameters: {str(e)}")
            raise constants.physics_parameter_error(f"Parameter setting failed: {str(e)}")
    
    def _set_network_inputs(self, input_values: np.ndarray) -> None:
        """
        Set input layer values in the neural network with input transformations.
        
        Transforms (a, b) → (1/a, 1/b) for better numerical scaling.
        
        Args:
            input_values (np.ndarray): Input parameter values
            
        Raises:
            constants.matrix_operation_error: If input dimensions don't match
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network available")
        
        input_layer = self.current_network.layers[0]
        
        if len(input_values) != len(input_layer.neurons):
            raise constants.matrix_operation_error(
                f"Input size mismatch: got {len(input_values)}, expected {len(input_layer.neurons)}"
            )
        
        # Apply input transformations for better scaling
        transformed_inputs = input_values.copy()
        
        # Transform (a, b) → (1/a, 1/b) to put integer parameters in (0,1] range
        if len(transformed_inputs) >= 2:
            a, b = transformed_inputs[0], transformed_inputs[1]
            if a != 0 and b != 0:  # Avoid division by zero
                transformed_inputs[0] = 1.0 / float(a)
                transformed_inputs[1] = 1.0 / float(b)
        
        for i, neuron in enumerate(input_layer.neurons):
            if isinstance(neuron, dummy_neuron):
                neuron.output = transformed_inputs[i]
    
    def get_network_info(self) -> dict:
        """
        Get information about the current network architecture.
        
        Returns:
            dict: Network architecture information
        """
        if self.current_network is None:
            return {"status": "no_network_built"}
        
        layer_sizes = [len(layer.neurons) for layer in self.current_network.layers]
        total_params = 0
        
        # Count parameters (weights)
        for layer in self.current_network.layers[1:]:  # Skip input layer
            for neuron in layer.neurons:
                total_params += len(neuron.inputs)
        
        info = {
            "status": "network_built",
            "layer_sizes": layer_sizes,
            "total_parameters": total_params,
            "network_config": self.network_config.copy(),
            "has_tbg_params": self.current_tbg_params is not None,
            "has_graphs": self.current_graph is not None
        }
        
        if self.current_tbg_params:
            info["current_tbg_params"] = self.current_tbg_params.copy()
            
        return info
    
    def reset(self) -> None:
        """Reset builder state, clearing current network and parameters."""
        self.current_network = None
        self.current_tbg_params = None  
        self.current_graph = None
        self.current_periodic_graph = None
        logger.info("dirac_network_builder state reset")


# Convenience functions following project patterns
def build_default_dirac_network() -> Tuple[dirac_network_builder, neural_network]:
    """
    Build a Dirac network with default configuration.
    
    Returns:
        Tuple[dirac_network_builder, neural_network]: Builder instance and constructed network
        
    Raises:
        constants.physics_parameter_error: If network construction fails
    """
    builder = dirac_network_builder()
    network = builder.build_network()
    return builder, network


def validate_dirac_parameters(params: List[Union[int, float]]) -> dict:
    """
    Validate Dirac point prediction parameters using project utilities.
    
    Convenience function that doesn't require a builder instance.
    
    Args:
        params (List[Union[int, float]]): TBG parameters to validate
        
    Returns:
        dict: Validated parameter dictionary
        
    Raises:
        constants.physics_parameter_error: If parameters are invalid
    """
    builder = dirac_network_builder()
    return builder.validate_tbg_parameters(params)