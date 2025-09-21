"""
Neural Network Base Infrastructure for TBG Project.

This module contains the fundamental neural network classes that provide
the core infrastructure for building neural networks in the TBG project.
These classes are parameter-agnostic and can be used for various applications.

Classes:
    dummy_neuron: Input layer neurons with fixed values
    neuron: Computational neurons with activation functions
    layer: Collections of neurons forming network layers
    neural_network: Complete neural network with forward/backward propagation
"""

import constants
from constants import np, logging
from constants import List, Tuple, Callable, Optional, Union, Any
from Generate_training_data import gradient_decent_adam

# Configure logging
logger = logging.getLogger(__name__)

# Activation functions
def tanh_activation(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Hyperbolic tangent activation function for hidden layers.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply tanh activation.
        
    Returns:
        Union[float, np.ndarray]: tanh(x) - output in range (-1, 1).
    """
    return np.tanh(x)


def tanh_activation_der(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of the hyperbolic tangent activation function.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate the derivative.
        
    Returns:
        Union[float, np.ndarray]: 1 - tanh²(x) - always positive, no dead zones.
    """
    return 1 - np.tanh(x)**2


def scaled_sigmoid_velocity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Scaled sigmoid activation function for velocity output (0-10 range).
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply activation.
        
    Returns:
        Union[float, np.ndarray]: 10 / (1 + exp(-x)) - output in range (0, 10).
    """
    # Clip x to prevent overflow: for x > max, exp(-x) ≈ 0, sigmoid ≈ 10
    x_clipped = np.clip(x, constants.SIGMOID_CLIPPING_RANGE[0], constants.SIGMOID_CLIPPING_RANGE[1])
    return constants.SCALED_SIGMOID_MAX / (1.0 + np.exp(-x_clipped))


def scaled_sigmoid_velocity_der(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of scaled sigmoid activation function for velocity.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate derivative.
        
    Returns:
        Union[float, np.ndarray]: Derivative of scaled sigmoid - always positive.
    """
    # Clip x to prevent overflow: use stable sigmoid derivative formula
    x_clipped = np.clip(x, constants.SIGMOID_CLIPPING_RANGE[0], constants.SIGMOID_CLIPPING_RANGE[1])  # Conservative clipping
    # For large positive x: sigmoid ≈ 1, derivative ≈ 0
    # For large negative x: sigmoid ≈ 0, derivative ≈ 0
    sigmoid_val = 1.0 / (1.0 + np.exp(-x_clipped))
    return constants.SCALED_SIGMOID_MAX * sigmoid_val * (1.0 - sigmoid_val)


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    ReLU activation function for hidden layers.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply ReLU activation.
        
    Returns:
        Union[float, np.ndarray]: max(0, x)
    """
    return np.maximum(x, 0)


def relu_der(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of ReLU activation function.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate derivative.
        
    Returns:
        Union[float, np.ndarray]: 1 if x > 0, 0 otherwise
    """
    return np.where(x > 0, 1, 0)


def leaky_relu(x: Union[float, np.ndarray], alpha: float = constants.LEAKY_RELU_ALPHA) -> Union[float, np.ndarray]:
    """
    Leaky ReLU activation function to prevent dead neurons.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply activation.
        alpha (float): Slope for negative values. Defaults to 0.01.
        
    Returns:
        Union[float, np.ndarray]: x if x > 0, alpha*x otherwise
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_der(x: Union[float, np.ndarray], alpha: float = constants.LEAKY_RELU_ALPHA) -> Union[float, np.ndarray]:
    """
    Derivative of Leaky ReLU activation function.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate derivative.
        alpha (float): Slope for negative values. Defaults to 0.01.
        
    Returns:
        Union[float, np.ndarray]: 1 if x > 0, alpha otherwise
    """
    return np.where(x > 0, 1, alpha)


def log_bounded_activation(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Log-bounded activation function for handling large positive inputs.
    
    Linear for small inputs (x < 10), logarithmic growth for large inputs.
    Prevents gradient explosion from unbounded positive values like (a,b) integers.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply activation.
        
    Returns:
        Union[float, np.ndarray]: f(x) = x if x < 10 else 10 + log(max(1 + x - 10, 1e-8))
    """
    threshold = constants.LOG_ACTIVATION_THRESHOLD
    # Ensure we don't get negative or zero arguments to log
    log_arg = np.maximum(1.0 + x - threshold, constants.LOG_ACTIVATION_MIN_ARG)
    return np.where(x < threshold, x, threshold + np.log(log_arg))


def log_bounded_activation_der(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of log-bounded activation function.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate derivative.
        
    Returns:
        Union[float, np.ndarray]: f'(x) = 1 if x < 10 else 1/max(1 + x - 10, 1e-8)
    """
    threshold = constants.LOG_ACTIVATION_THRESHOLD
    # Ensure we don't divide by zero or negative values
    denominator = np.maximum(1.0 + x - threshold, constants.LOG_ACTIVATION_MIN_ARG)
    return np.where(x < threshold, 1.0, 1.0 / denominator)



def wrapped_k_activation(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wrapped activation function for k-points in Brillouin zone.
    
    Applies ReLU followed by modular wrapping to [-0.5, 0.5] range.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) to apply activation.
        
    Returns:
        Union[float, np.ndarray]: Wrapped k-point values in canonical range.
    """
    # First apply leaky ReLU to prevent dead neurons
    activated = leaky_relu(x)
    # Then wrap to [-0.5, 0.5] range (canonical Brillouin zone)
    return ((activated + 0.5) % 1.0) - 0.5


def wrapped_k_activation_der(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of wrapped k-point activation function.
    
    Since wrapping preserves derivatives (modular arithmetic), 
    this is just the leaky ReLU derivative.
    
    Args:
        x (Union[float, np.ndarray]): Input value(s) at which to evaluate derivative.
        
    Returns:
        Union[float, np.ndarray]: 1 if x > 0, 0.01 otherwise (leaky ReLU derivative)
    """
    return leaky_relu_der(x)


class dummy_neuron:
    """
    A dummy neuron that holds a fixed output value without computation.
    Used for input layers where values are externally provided.
    
    Attributes:
        position (np.ndarray): 2D position (i, layer_index) of the neuron in the network.
        output (Any): Fixed output value that this neuron holds.
    """
    def __init__(self, position: Tuple[int, int], output: Any) -> None:
        """
        Initialize a dummy neuron with a fixed output.
        
        Args:
            position (Tuple[int, int]): Position (neuron_index, layer_index) in the network.
            output (Any): The fixed output value this neuron will hold.
        """
        self.position = np.array(position)
        self.output = output

    def compute(self) -> None:
        """
        Placeholder compute method for interface compatibility.
        Dummy neurons do not perform any computation.
        """
        pass


class neuron:
    """
    A computational neuron that applies an activation function to weighted inputs.
    
    Attributes:
        position (np.ndarray): 2D position (i, layer_index) of the neuron in the network.
        activation_function (Callable): Function to apply to the weighted sum of inputs.
        activation_function_der (Callable): Derivative of the activation function.
        inputs (List[Tuple["neuron", float]]): List of (input_neuron, weight) pairs.
        output (Optional[Any]): Current output value of the neuron.
        sum (float): Weighted sum of inputs before activation.
        Adam_corrector (List[gradient_decent_adam]): ADAM optimizers for each input weight.
    """
    def __init__(self, position: Tuple[int, int], activation_function: Callable, 
                 activation_function_der: Callable, output: Optional[Any] = None) -> None:
        """
        Initialize a computational neuron.
        
        Args:
            position (Tuple[int, int]): Position (neuron_index, layer_index) in the network.
            activation_function (Callable): Activation function to apply to inputs.
            activation_function_der (Callable): Derivative of the activation function.
            output (Optional[Any], optional): Initial output value. Defaults to None.
        """
        self.position = np.array(position)
        self.activation_function = activation_function
        self.activation_function_der = activation_function_der
        self.inputs = []
        self.output = output
        self.sum = 0
        self.Adam_corrector = []

    def connect_input(self, other_neuron: "neuron", weight: float) -> None:
        """
        Connect another neuron as an input to this neuron with a specified weight.
        
        Args:
            other_neuron (neuron): The neuron to connect as input.
            weight (float): The connection weight between neurons.
        """
        self.inputs.append((other_neuron, weight))
        self.Adam_corrector.append(gradient_decent_adam(initial_weight=weight))

    def compute(self) -> None:
        """
        Compute the neuron's output by applying activation function to weighted inputs.
        Updates self.sum with the weighted sum and self.output with the activated result.
        """
        if not self.inputs:
            return
        self.sum = sum(weight * other_neuron.output for other_neuron, weight in self.inputs)
        self.output = self.activation_function(self.sum)

    def derivative(self, other_neuron: "neuron") -> float:
        """
        Compute the derivative of this neuron's output with respect to another neuron's input.
        
        Args:
            other_neuron (neuron): The input neuron to compute derivative with respect to.
            
        Returns:
            float: Derivative value (weight * activation_derivative) or 0 if not connected.
        """
        index = next(i for i, (neuron, _) in enumerate(self.inputs) if neuron == other_neuron)
        if index != -1:
            weight = self.inputs[index][1]
            return weight * self.activation_function_der(self.sum)
        return 0

    def change_weight(self, other_neuron: "neuron", new_weight: float) -> None:
        """
        Update the connection weight to a specific input neuron.
        
        Args:
            other_neuron (neuron): The input neuron whose connection weight to update.
            new_weight (float): The new weight value for the connection.
        """
        for i, (neuron, _) in enumerate(self.inputs):
            if neuron == other_neuron:
                self.inputs[i] = (neuron, new_weight)
                break


class layer:
    """
    A layer of neurons in the neural network.
    
    Attributes:
        index_layer (int): Index of this layer in the network.
        num_neurons (int): Number of neurons in this layer.
        Jacobian (List[np.ndarray]): Jacobian matrix rows for backpropagation.
        neurons (List[Union[neuron, dummy_neuron]]): List of neurons in this layer.
        activation_function (Optional[Callable]): Activation function for this layer.
        activation_function_der (Optional[Callable]): Derivative of activation function.
        Dummy (bool): Whether this is a dummy (input) layer.
        activation_function_der_matrix (np.ndarray): Diagonal matrix of activation derivatives.
    """
    def __init__(self, num_neurons: int, activation_function: Optional[Callable] = None,
                 activation_function_der: Optional[Callable] = None, index_layer: int = 0,
                 output: Optional[List[Any]] = None) -> None:
        """
        Initialize a layer with the specified number of neurons.
        
        Args:
            num_neurons (int): Number of neurons to create in this layer.
            activation_function (Optional[Callable], optional): Activation function for neurons.
            activation_function_der (Optional[Callable], optional): Derivative of activation function.
            index_layer (int, optional): Index of this layer in the network. Defaults to 0.
            output (Optional[List[Any]], optional): Fixed outputs for dummy layer. Defaults to None.
            
        Raises:
            ValueError: If output length doesn't match num_neurons for input layer.
        """
        self.index_layer = index_layer
        self.num_neurons = num_neurons
        self.Jacobian = []
        
        if output is not None:
            if len(output) != num_neurons:
                raise ValueError("Output length must match the number of neurons in the input layer.")
            self.neurons = [dummy_neuron((i, index_layer), output[i]) for i in range(num_neurons)]
            self.activation_function = None
            self.activation_function_der = None
            self.Dummy = True
        else:
            self.neurons = [neuron((i, index_layer), activation_function, activation_function_der) for i in range(num_neurons)]
            self.activation_function = activation_function
            self.activation_function_der = activation_function_der
            self.Dummy = False
        
        self.activation_function_der_matrix = np.zeros([num_neurons, num_neurons])

    def connect_layers(self, next_layer: "layer", weights: List[List[float]]) -> None:
        """
        Connect this layer to the next layer with specified weights.
        
        Args:
            next_layer (layer): The layer to connect to.
            weights (List[List[float]]): Weight matrix [i][j] from neuron i to neuron j.
            
        Raises:
            ValueError: If Jacobian is already initialized.
        """
        if len(self.Jacobian) != 0:
            raise ValueError("The Jacobian is already initialized!")
        
        for i, neuron in enumerate(self.neurons):
            for j, next_neuron in enumerate(next_layer.neurons):
                next_neuron.connect_input(neuron, weights[i][j])
            self.Jacobian.append(np.ones(len(next_layer.neurons)))

    def update_Jacobian(self, gradient: np.array) -> None:
        """
        Update the Jacobian matrix with new gradient information.
        
        Args:
            gradient (np.ndarray): Gradient array to update Jacobian with.
                Can be 1D or 2D array.
        """
        # Ensure Jacobian is properly sized
        if len(self.Jacobian) != self.num_neurons:
            logger.warning(f"Jacobian size mismatch: expected {self.num_neurons}, got {len(self.Jacobian)}")
            # Resize Jacobian if needed
            while len(self.Jacobian) < self.num_neurons:
                self.Jacobian.append(np.zeros(1))  # Default size, will be updated
        
        if gradient.ndim == 2:
            for i in range(self.num_neurons):
                if i < gradient.shape[0]:
                    self.Jacobian[i] = gradient[i, :] 
        else: 
            logger.warning("Using 1d Gradients")
            for i in range(self.num_neurons):
                if i < len(gradient):
                    self.Jacobian[i] = gradient[i] 

    def compute(self) -> None:
        """
        Compute outputs for all neurons in this layer.
        Also updates the activation function derivative matrix.
        """
        for i, neuron in enumerate(self.neurons):
            neuron.compute()
            # Only compute activation derivative for real neurons, not dummy neurons
            if hasattr(neuron, 'activation_function_der') and hasattr(neuron, 'sum'):
                self.activation_function_der_matrix[i][i] = neuron.activation_function_der(neuron.sum)
            else:
                # For dummy neurons, derivative is 1 (identity function)
                self.activation_function_der_matrix[i][i] = 1.0

    def update_weights(self) -> None:
        """
        Update connection weights using ADAM optimizer.
        Only applies to non-dummy layers.
        """
        if self.Dummy:
            return
        
        for i, neuron in enumerate(self.neurons):
            for j, input in enumerate(neuron.inputs):
                neuron.Adam_corrector[j].update(self.Jacobian[i][j])  # compute the new weight
                neuron.change_weight(input[0], neuron.Adam_corrector[j].weight)  # FIXED: input[0] is the neuron 

    def backward(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform backward pass computation for this layer.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Jacobian matrix (transposed) for gradient computation
                - Values array (transposed) containing neuron outputs
        """
        values = []
        
        if self.Dummy:
            for neuron in self.neurons:
                values.append(neuron.output)
            values_array = np.array(values)
            # If this is a dummy layer, we don't compute gradients
            return np.identity(len(self.neurons)), values_array.T
        
        # Compute Jacobian matrix for non-dummy layers
        Jacobian = []
        for neuron in self.neurons:
            gradient_temp = []
            for other_neuron, _ in neuron.inputs:
                gradient_temp.append(neuron.derivative(other_neuron))
            Jacobian.append(gradient_temp)
            values.append(neuron.output)
        
        # Convert the list of gradients and values to a numpy array
        Jacobian_array = np.array(Jacobian)
        values_array = np.array(values)

        return Jacobian_array.T, values_array.T


class neural_network:
    """
    A feedforward neural network with customizable architecture.
    
    Attributes:
        layers (List[layer]): List of layers in the network.
        loss_function_and_grad (Optional[Callable]): Loss function that returns loss and gradients.
    """
    def __init__(self, layers_structure: List[Tuple[int, Callable, Callable]], 
                 inputs: Optional[List[Any]] = None, 
                 loss_function_and_grad: Optional[Callable] = None) -> None:
        """
        Initialize a neural network with specified architecture.
        
        Args:
            layers_structure (List[Tuple[int, Callable, Callable]]): 
                List of (num_neurons, activation_func, activation_deriv) for each layer.
            inputs (Optional[List[Any]], optional): Input values for the input layer. 
                Defaults to [(0, 0)].
            loss_function_and_grad (Optional[Callable], optional): 
                Function that computes loss and gradients. Defaults to None.
        """
        self.layers = []
        self.loss_function_and_grad = loss_function_and_grad
        
        if inputs is None:
            inputs = [(0, 0)]
        
        # Create input layer
        input_layer = layer(len(inputs), lambda x: x, lambda x: 1, index_layer=0, output=inputs)  # Identity activation for input layer
        self.layers.append(input_layer)
        
        # Create hidden and output layers
        for index, (num_neurons, activation_function, activation_function_gradient) in enumerate(layers_structure):
            # Handle mixed activation functions for output layer
            if activation_function == 'mixed_output':
                new_layer = self._create_mixed_output_layer(num_neurons, index+1)
            else:
                new_layer = layer(num_neurons, activation_function, activation_function_gradient, index_layer=index+1)
            
            # Connect to the previous layer with He normal initialization BEFORE appending new layer
            previous_layer = self.layers[index]  # index is correct since we haven't appended yet
            n_inputs = len(previous_layer.neurons)
            
            # He initialization for ReLU-like activations: std = sqrt(HE_FACTOR/n_inputs)
            weights = np.random.normal(0.0, np.sqrt(constants.HE_INIT_FACTOR / n_inputs), (n_inputs, num_neurons))
            
            # He initialization is optimal for ReLU/Leaky ReLU to prevent gradient explosion
            
            previous_layer.connect_layers(new_layer, weights)
            self.layers.append(new_layer)

    def _create_mixed_output_layer(self, num_neurons: int, layer_index: int) -> 'layer':
        """
        Create a mixed output layer with different activation functions per neuron.
        First 2 neurons use wrapped k-point activation, remaining use ReLU for ν output.
        
        Args:
            num_neurons: Number of neurons in the layer
            layer_index: Index of this layer in the network
            
        Returns:
            Layer with mixed activation functions
        """
        # Create layer without specifying activation (we'll set per-neuron)
        mixed_layer = layer(num_neurons, None, None, layer_index)
        
        # Override the neurons with custom activation functions
        mixed_layer.neurons = []
        for i in range(num_neurons):
            if i < 2:  # First 2 neurons are k_x, k_y - use wrapped activation
                neuron_obj = neuron((i, layer_index), wrapped_k_activation, wrapped_k_activation_der)
            else:  # Remaining neurons (ν = 1/(1+v)) use Leaky ReLU
                neuron_obj = neuron((i, layer_index), leaky_relu, leaky_relu_der)
            mixed_layer.neurons.append(neuron_obj)
        
        mixed_layer.Dummy = False
        return mixed_layer

    def compute(self) -> List[Any]:
        """
        Perform forward pass through the network.
        
        Returns:
            List[Any]: Output values from the final layer.
        """
        for layer_obj in self.layers:
            layer_obj.compute()
        # After computing all layers, the output will be in the last layer's neurons
        return [neuron.output for neuron in self.layers[-1].neurons]

    def backward(self) -> None:
        """
        Perform backward pass through the network using backpropagation.
        Updates Jacobian matrices for all layers based on loss function gradients.
        """
        output = self.compute()
        loss, gradient_vector = self.loss_function_and_grad(output, [0.6, 0.3, 0.1])  # Default weights

        Jacobians = []
        activation_der_matrices = []
        values = []
        
        # Collect gradients from all layers
        for layer in self.layers:
            Jac_temp, values_temp = layer.backward()
            Jacobians.append(Jac_temp)
            activation_der_matrices.append(layer.activation_function_der_matrix)
            values.append(values_temp)

        # Backpropagate gradients through the network
        current_vector = gradient_vector
        Temp_vector = current_vector * np.diag(activation_der_matrices[-1])
        full_Jacobian = np.outer(Temp_vector,values[-2])  
        self.layers[-1].update_Jacobian(full_Jacobian)  
        for i in range(len(Jacobians)-2, 0, -1):
            current_vector = current_vector @ Jacobians[i+1].T
            Temp_vector = current_vector * np.diag(activation_der_matrices[i])
            full_Jacobian = (np.outer(Temp_vector,values[i-1])) 
            self.layers[i].update_Jacobian(full_Jacobian)  

    def update_weights(self) -> None:
        """
        Update all network weights using computed gradients.
        Calls backward() to compute gradients, then updates weights in all layers.
        """
        self.backward()
        for layer_obj in self.layers:
            layer_obj.update_weights()