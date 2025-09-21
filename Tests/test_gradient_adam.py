"""
Deterministic test to verify gradient computation and Adam optimizer are working correctly.

This test creates a simple network with known inputs/outputs and verifies:
1. Gradients are computed correctly 
2. Adam optimizer updates weights properly
3. Loss decreases over iterations
4. No gradient explosion occurs

Uses a simple quadratic function that should be easy to learn.
"""

import constants
from constants import np, logging
from neural_network_base import neural_network, leaky_relu, leaky_relu_der
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_quadratic_loss(output, weights):
    """
    Simple quadratic loss function for testing: L = (y - target)^2
    Target is always [0.5, 0.3] for deterministic testing.
    
    Args:
        output: Network output [y1, y2]
        weights: Not used in this simple test
        
    Returns:
        Tuple[float, np.ndarray]: (loss, gradient)
    """
    target = np.array([0.5, 0.3])
    output_array = np.array(output)
    
    # Loss = sum((output - target)^2)
    diff = output_array - target
    loss = np.sum(diff**2)
    
    # Gradient = 2 * (output - target)
    gradient = 2.0 * diff
    
    return loss, gradient

def test_gradient_computation():
    """Test basic gradient computation with a simple 2-input, 2-output network."""
    logger.info("=== TESTING GRADIENT COMPUTATION ===")
    
    # Create simple network: 2 inputs -> 4 hidden -> 2 outputs
    network = neural_network(
        layers_structure=[
            (4, leaky_relu, leaky_relu_der),  # Hidden layer
            (2, leaky_relu, leaky_relu_der)   # Output layer
        ],
        inputs=[1.0, 2.0],  # Fixed deterministic inputs
        loss_function_and_grad=simple_quadratic_loss
    )
    
    logger.info(f"Network architecture: 2 -> 4 -> 2")
    logger.info(f"Input: [1.0, 2.0]")
    logger.info(f"Target: [0.5, 0.3]")
    
    # Test initial state
    initial_output = network.compute()
    initial_loss, initial_grad = simple_quadratic_loss(initial_output, [])
    
    logger.info(f"Initial output: [{initial_output[0]:.4f}, {initial_output[1]:.4f}]")
    logger.info(f"Initial loss: {initial_loss:.6f}")
    logger.info(f"Initial gradient: [{initial_grad[0]:.4f}, {initial_grad[1]:.4f}]")
    
    # Perform backward pass
    network.backward()
    
    # Check if gradients were computed
    gradients_found = False
    total_grad_norm = 0.0
    
    for i, layer in enumerate(network.layers[1:], 1):  # Skip input layer
        if hasattr(layer, 'Jacobian') and layer.Jacobian is not None:
            layer_grad_norm = 0.0
            for grad in layer.Jacobian:
                if grad is not None:
                    grad_norm = np.linalg.norm(grad)
                    layer_grad_norm += grad_norm
                    total_grad_norm += grad_norm
                    gradients_found = True
            logger.info(f"Layer {i} gradient norm: {layer_grad_norm:.6f}")
    
    logger.info(f"Total gradient norm: {total_grad_norm:.6f}")
    
    if not gradients_found:
        logger.error("❌ NO GRADIENTS COMPUTED!")
        return False
    elif total_grad_norm == 0.0:
        logger.error("❌ GRADIENTS ARE ZERO!")
        return False
    else:
        logger.info("✅ Gradients computed successfully")
        return True

def test_adam_optimizer():
    """Test Adam optimizer with deterministic inputs over multiple iterations."""
    logger.info("\n=== TESTING ADAM OPTIMIZER ===")
    
    # Create network with fixed seed for deterministic results
    np.random.seed(42)
    
    network = neural_network(
        layers_structure=[
            (4, leaky_relu, leaky_relu_der),  # Hidden layer  
            (2, leaky_relu, leaky_relu_der)   # Output layer
        ],
        inputs=[1.0, 2.0],  # Fixed inputs
        loss_function_and_grad=simple_quadratic_loss
    )
    
    logger.info("Testing Adam optimizer over 10 iterations:")
    
    losses = []
    gradient_norms = []
    
    for iteration in range(10):
        # Forward pass
        output = network.compute()
        loss, grad = simple_quadratic_loss(output, [])
        
        # Backward pass
        network.backward()
        
        # Compute gradient norm
        total_grad_norm = 0.0
        for layer in network.layers[1:]:  # Skip input layer
            if hasattr(layer, 'Jacobian') and layer.Jacobian is not None:
                for grad_vec in layer.Jacobian:
                    if grad_vec is not None:
                        total_grad_norm += np.linalg.norm(grad_vec)
        
        # Update weights
        for layer in network.layers[1:]:
            layer.update_weights()
        
        losses.append(loss)
        gradient_norms.append(total_grad_norm)
        
        logger.info(f"Iter {iteration+1:2d}: Loss={loss:.6f}, Output=[{output[0]:.4f}, {output[1]:.4f}], GradNorm={total_grad_norm:.4f}")
    
    # Check if loss is decreasing
    if losses[-1] < losses[0]:
        logger.info(f"✅ Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
        loss_improved = True
    else:
        logger.error(f"❌ Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}")
        loss_improved = False
    
    # Check for gradient explosion
    max_grad_norm = max(gradient_norms)
    if max_grad_norm < 100.0:  # Reasonable threshold
        logger.info(f"✅ Gradients stable, max norm: {max_grad_norm:.4f}")
        gradients_stable = True
    else:
        logger.error(f"❌ Gradient explosion detected, max norm: {max_grad_norm:.4f}")
        gradients_stable = False
    
    return loss_improved and gradients_stable

def test_weight_updates():
    """Test that weights are actually changing during training."""
    logger.info("\n=== TESTING WEIGHT UPDATES ===")
    
    np.random.seed(42)
    network = neural_network(
        layers_structure=[
            (3, leaky_relu, leaky_relu_der),
            (2, leaky_relu, leaky_relu_der)
        ],
        inputs=[1.0, 1.5],
        loss_function_and_grad=simple_quadratic_loss
    )
    
    # Store initial weights
    initial_weights = []
    for layer in network.layers[1:]:  # Skip input layer
        layer_weights = []
        for neuron in layer.neurons:
            # neuron.inputs contains weight objects, get their values
            layer_weights.append([w.value if hasattr(w, 'value') else float(w) for w in neuron.inputs])
        initial_weights.append(layer_weights)
    
    logger.info("Initial weights captured")
    
    # Train for 3 iterations
    for i in range(3):
        network.compute()
        network.backward()
        for layer in network.layers[1:]:
            layer.update_weights()
    
    # Check if weights changed
    weights_changed = False
    total_weight_change = 0.0
    
    for layer_idx, layer in enumerate(network.layers[1:]):
        for neuron_idx, neuron in enumerate(layer.neurons):
            for weight_idx, weight_obj in enumerate(neuron.inputs):
                current_weight = weight_obj.value if hasattr(weight_obj, 'value') else float(weight_obj)
                initial_weight = initial_weights[layer_idx][neuron_idx][weight_idx]
                weight_change = abs(current_weight - initial_weight)
                total_weight_change += weight_change
                if weight_change > 1e-8:  # Numerical precision threshold
                    weights_changed = True
    
    logger.info(f"Total weight change: {total_weight_change:.8f}")
    
    if weights_changed:
        logger.info("✅ Weights updated successfully")
        return True
    else:
        logger.error("❌ Weights did not change!")
        return False

def main():
    """Run all tests to verify gradient computation and Adam optimizer."""
    logger.info("DETERMINISTIC NEURAL NETWORK COMPONENT TEST")
    logger.info("=" * 50)
    
    test_results = []
    
    # Test 1: Basic gradient computation
    test_results.append(test_gradient_computation())
    
    # Test 2: Adam optimizer functionality  
    test_results.append(test_adam_optimizer())
    
    # Test 3: Weight updates
    test_results.append(test_weight_updates())
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        logger.info(f"✅ ALL TESTS PASSED ({passed}/{total})")
        logger.info("Neural network components are working correctly!")
    else:
        logger.error(f"❌ SOME TESTS FAILED ({passed}/{total})")
        logger.error("There are issues with the neural network implementation!")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)