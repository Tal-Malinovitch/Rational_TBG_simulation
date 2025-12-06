"""
Training workflows for Dirac point prediction neural networks.

This module handles all training-related functionality including pretraining,
hybrid training, general training, and loss functions. Uses existing project
utilities like simulation_data_loader for data management.

Classes:
    dirac_network_trainer: Handles all neural network training workflows
"""

import constants
from constants import np, os, time, glob, logging
from constants import List, Tuple, Dict, Optional, Union, Any
from simulation_data_loader import simulation_data_analyzer
from stats import statistics
from TBG import Dirac_analysis
from neural_network_base import neural_network, dummy_neuron
from preprocess_training_data import preprocess_training_data, load_preprocessed_data

# Configure logging
logger = logging.getLogger(__name__)

# Default training configuration
default_training_config = {
    'learning_rate': constants.DEFAULT_LEARNING_RATE,
    'batch_size': constants.DEFAULT_TRAINING_BATCH_SIZE,
    'validation_split': constants.DEFAULT_VALIDATION_SPLIT,
    'early_stopping_patience': constants.DEFAULT_EARLY_STOPPING_PATIENCE,
    'checkpoint_interval': constants.DEFAULT_CHECKPOINT_INTERVAL,
    'max_epochs': constants.DEFAULT_MAX_EPOCHS,
    'gradient_clip_value': constants.GRADIENT_CLIP_VALUE,
    'loss_explosion_threshold': constants.LOSS_EXPLOSION_THRESHOLD,
    'learning_rate_reduction_factor': constants.LEARNING_RATE_REDUCTION_FACTOR,
    'min_learning_rate': constants.MIN_LEARNING_RATE,
    'loss_history_window': constants.LOSS_HISTORY_WINDOW
}

# Default loss weights for physics-based training
default_loss_weights = constants.DEFAULT_NN_LOSS_WEIGHTS


class dirac_network_trainer:
    """
    Handles training workflows for Dirac point prediction neural networks.
    
    This class manages different training strategies including data-only pretraining,
    physics-based optimization, and hybrid approaches. Integrates with existing
    project utilities for data loading and statistics tracking.
    
    Attributes:
        training_config (dict): Training configuration parameters
        stats (statistics): Statistics tracking using project utility
        current_network (Optional[neural_network]): Network being trained
        current_network_builder (Optional): Associated network builder
        training_start_time (Optional[float]): Training session start time
    """
    
    def __init__(self, training_config: Optional[dict] = None) -> None:
        """
        Initialize the Dirac network trainer.
        
        Args:
            training_config (dict, optional): Training configuration parameters.
                Uses default_training_config if not provided.
                
        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        try:
            self.training_config = training_config or default_training_config.copy()
            self._validate_training_config()
            
            # Initialize statistics using project utility
            self.stats = statistics()
            self.training_start_time: Optional[float] = None
            
            # Training state
            self.current_network: Optional[neural_network] = None
            self.current_network_builder: Optional = None
            self._current_target: Optional[np.ndarray] = None
            self.persistence: Optional = None  # Will be set by orchestrator

            # Adaptive learning rate and stability tracking
            self.current_learning_rate: float = self.training_config['learning_rate']
            self.loss_history: List[float] = []
            self.learning_rate_reductions: int = 0

            # ν clamping statistics tracking
            self.nu_clamp_count = 0
            self.batch_sample_count = 0
            
            logger.info(f"dirac_network_trainer initialized with config: {self.training_config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dirac_network_trainer: {str(e)}")
            raise constants.physics_parameter_error(f"Trainer initialization failed: {str(e)}")
    
    def _validate_training_config(self) -> None:
        """
        Validate training configuration parameters.
        
        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        required_keys = ['learning_rate', 'batch_size', 'validation_split', 'early_stopping_patience']
        
        for key in required_keys:
            if key not in self.training_config:
                raise constants.physics_parameter_error(f"Missing required training config key: {key}")
        
        # Validate positive numeric values
        if self.training_config['learning_rate'] <= 0:
            raise constants.physics_parameter_error("learning_rate must be positive")
        if self.training_config['batch_size'] <= 0:
            raise constants.physics_parameter_error("batch_size must be positive")
        if not 0 < self.training_config['validation_split'] < 1:
            raise constants.physics_parameter_error("validation_split must be between 0 and 1")
    
    def set_network(self, network: neural_network, network_builder=None) -> None:
        """
        Set the network to be trained.
        
        Args:
            network (neural_network): Neural network to train
            network_builder (optional): Associated network builder for parameter setting
        """
        self.current_network = network
        self.current_network_builder = network_builder
        logger.info("Set network for training")
    
    def set_persistence(self, persistence) -> None:
        """
        Set the persistence manager for checkpoint saving.
        
        Args:
            persistence: Persistence manager instance
        """
        self.persistence = persistence
        logger.info("Set persistence manager for checkpoint saving")
    
    def load_training_data(self, data_folder: Optional[str] = None, 
                          remove_duplicates: bool = False) -> simulation_data_analyzer:
        """
        Load training data using project's simulation_data_analyzer.
        
        Args:
            data_folder (str, optional): Path to training data folder
            remove_duplicates (bool): Whether to remove duplicate data points
            
        Returns:
            simulation_data_analyzer: Loaded data analyzer
            
        Raises:
            ValueError: If no training data found
        """
        if data_folder is None:
            data_folder = os.path.join(constants.PATH, "Training_data")
        
        try:
            # Use existing project utility
            analyzer = simulation_data_analyzer(data_folder)
            
            # Find CSV files
            csv_pattern = os.path.join(data_folder, "*.csv")
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {data_folder}")
            
            # Load using existing utility
            csv_filenames = [os.path.basename(f) for f in csv_files]
            analyzer.load_multiple_csv_files(csv_filenames, remove_duplicates=remove_duplicates)
            
            if not analyzer.data_points:
                raise ValueError("No valid training data loaded")
            
            logger.info(f"Loaded {len(analyzer.data_points)} training examples from {len(csv_filenames)} files")
            
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def prepare_training_data(self, data_points: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from simulation results with transformations.
        
        Applies input transform: (a,b) → (1/a, 1/b)
        Applies output transform: v → ν = 1/(1+v)
        
        Args:
            data_points: List of simulation_data_point objects
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (transformed_inputs, transformed_targets) for training
        """
        inputs = []
        targets = []
        
        for dp in data_points:
            # Input features with transformation: (a,b) → (1/a, 1/b)
            a, b = dp.a, dp.b
            if a != 0 and b != 0:  # Avoid division by zero
                input_features = [
                    1.0/float(a), 1.0/float(b),  # Transformed (a,b)
                    dp.interlayer_dist_threshold, dp.intralayer_dist_threshold,
                    dp.inter_graph_weight, dp.intra_graph_weight
                ]
            else:
                continue  # Skip invalid data points
            
            # Target outputs with velocity transform: v → ν = 1/(1+v)
            velocity = dp.velocity
            if velocity >= 0:  # Ensure valid velocity
                nu = 1.0 / (1.0 + velocity)
                target_output = [dp.k_x, dp.k_y, nu]  # k_x, k_y, ν
            else:
                continue  # Skip invalid velocities
            
            inputs.append(input_features)
            targets.append(target_output)
        
        logger.info(f"Prepared {len(inputs)} training examples with input/output transformations")
        return np.array(inputs), np.array(targets)
    
    def pretrain_from_data(self, data_folder: Optional[str] = None, epochs: int = 200,
                          batch_size: Optional[int] = None, validation_split: Optional[float] = None,
                          start_epoch: int = 0, resume_patience_counter: int = 0,
                          resume_best_loss: Optional[float] = None) -> dict:
        """
        Pretrain network on historical data using minimum distance loss.

        This method handles multi-valued Dirac point data by using a minimum distance
        loss function that compares the network output to the closest valid Dirac point
        for each parameter set, rather than averaging all targets.

        Args:
            data_folder (str, optional): Path to training data. Defaults to 'Training_data'.
            epochs (int): Number of training epochs
            batch_size (int, optional): Batch size, uses config default if None
            validation_split (float, optional): Validation split, uses config default if None
            start_epoch (int): Starting epoch number for resume (for display/logging). Defaults to 0.
            resume_patience_counter (int): Resume early stopping patience counter. Defaults to 0.
            resume_best_loss (float, optional): Best validation loss from checkpoint to resume. If None, starts with infinity.

        Returns:
            dict: Training results and statistics

        Raises:
            constants.physics_parameter_error: If no network set or training fails
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network set. Call set_network() first.")
        
        # Use config defaults if not specified
        batch_size = batch_size or self.training_config['batch_size']
        validation_split = validation_split or self.training_config['validation_split']
        
        try:
            logger.info("Starting pretraining phase with minimum distance loss...")
            self.training_start_time = time.time()
            self.stats.start_time = self.training_start_time

            # Use default data folder if not specified
            if data_folder is None:
                data_folder = 'Training_data'

            # Check if preprocessed data exists, if not run preprocessing
            preprocessed_path = os.path.join(data_folder, 'grouped_training_data.pkl')
            if not os.path.exists(preprocessed_path):
                logger.info("Preprocessed data not found. Running preprocessing...")
                input_csv = os.path.join(data_folder, 'dirac_training_data.csv')
                preprocess_training_data(input_csv, preprocessed_path)
                logger.info("Preprocessing complete!")

            # Load grouped data
            logger.info(f"Loading preprocessed data from {preprocessed_path}")
            data = load_preprocessed_data(preprocessed_path)
            training_samples = data['training_samples']
            logger.info(f"Loaded {len(training_samples)} parameter sets with {data['total_dirac_points']} total Dirac points")

            # Split data by parameter sets (not individual Dirac points)
            split_idx = int(len(training_samples) * (1 - validation_split))
            np.random.seed(42)  # Fixed seed for consistent train/val split
            indices = np.random.permutation(len(training_samples))
            np.random.seed()  # Reset seed for future randomness
            train_indices, val_indices = indices[:split_idx], indices[split_idx:]

            train_samples = [training_samples[i] for i in train_indices]
            val_samples = [training_samples[i] for i in val_indices]

            logger.info(f"Training set: {len(train_samples)} parameter sets")
            logger.info(f"Validation set: {len(val_samples)} parameter sets")
            
            # Log network architecture with parameter counts
            total_params = 0
            logger.info("=== NETWORK ARCHITECTURE ===")
            for i, layer in enumerate(self.current_network.layers):
                if layer.Dummy:
                    # Input layer - no trainable parameters
                    logger.info(f"Layer {i} (Input): {len(layer.neurons)} neurons, 0 parameters")
                else:
                    # Hidden/output layer - count weights and biases
                    # Each neuron has: len(inputs) weights + 1 bias
                    layer_params = sum(len(neuron.inputs) + 1 for neuron in layer.neurons)
                    total_params += layer_params
                    layer_type = "Hidden" if i < len(self.current_network.layers) - 1 else "Output"
                    logger.info(f"Layer {i} ({layer_type}): {len(layer.neurons)} neurons, {layer_params} parameters")
            logger.info(f"Total network parameters: {total_params}")
            logger.info("==========================")
            
            # Store original loss function and set minimum distance loss
            # Wrap loss function so backward() doesn't need to pass weights
            def wrapped_min_dist_loss(output, weights=None):
                # Ignore weights parameter from backward(), use proper supervised weights
                loss, gradients, _, _ = self._minimum_distance_loss_function(output, constants.DEFAULT_SUPERVISED_LOSS_WEIGHTS)
                return loss, gradients

            original_loss_func = self.current_network.loss_function_and_grad
            self.current_network.loss_function_and_grad = wrapped_min_dist_loss

            # Training loop with statistics tracking
            best_val_loss = resume_best_loss if resume_best_loss is not None else float('inf')
            patience_counter = resume_patience_counter
            training_losses = []
            validation_losses = []

            if resume_best_loss is not None:
                logger.info(f"Resuming training with best validation loss: {best_val_loss:.6f}")
            else:
                logger.info("Starting training from scratch (no previous best loss)")

            for epoch in range(epochs):
                epoch_start_time = time.time()

                # Shuffle and train with grouped data (with fixed seed for reproducibility)
                np.random.seed(constants.EPOCH_SHUFFLE_SEED + epoch)
                epoch_indices = np.random.permutation(len(train_samples))
                np.random.seed()  # Reset seed for other operations
                shuffled_train_samples = [train_samples[i] for i in epoch_indices]

                epoch_loss = self._train_epoch_grouped(shuffled_train_samples, batch_size)
                val_loss = self._evaluate_validation_loss_grouped(val_samples)
                
                training_losses.append(epoch_loss)
                validation_losses.append(val_loss)

                # Check for loss explosion and adapt learning rate
                if self._detect_loss_explosion(val_loss):
                    logger.warning(f"Loss explosion detected at epoch {epoch}")
                    if self._reduce_learning_rate():
                        logger.info("Learning rate reduced, continuing training...")
                        # Clear recent loss history to give new learning rate a chance
                        self.loss_history = self.loss_history[-2:]  # Keep only last 2 entries
                    else:
                        logger.error("Learning rate at minimum, cannot reduce further")
                        # Could potentially stop training here if desired

                # Log statistics
                epoch_duration = time.time() - epoch_start_time
                self.stats.log_combination(
                    duration=epoch_duration,
                    system_size=len(train_samples),
                    n_scale=self.current_learning_rate,  # Use current adaptive learning rate
                    success=(val_loss < best_val_loss),
                    no_intersection=True,
                    num_of_Dirac=1
                )
                
                # Progress reporting every N epochs
                if epoch % constants.EPOCH_LOGGING_FREQUENCY == 0:
                    # Monitor network weights for stability
                    first_weight = 0.0
                    try:
                        if (len(self.current_network.layers) > 1 and 
                            len(self.current_network.layers[1].neurons) > 0 and 
                            len(self.current_network.layers[1].neurons[0].inputs) > 0):
                            # inputs is a list of tuples (neuron, weight), so get the weight from tuple[1]
                            first_weight = self.current_network.layers[1].neurons[0].inputs[0][1]
                    except (IndexError, TypeError):
                        first_weight = 0.0
                    
                    actual_epoch = start_epoch + epoch + 1  # +1 because we display 1-indexed epochs
                    logger.info(f"Epoch {actual_epoch:3d}/{start_epoch + epochs}: Train Loss: {epoch_loss:.6f}, "
                              f"Val Loss: {val_loss:.6f}, Time: {epoch_duration:.1f}s, Weight: {first_weight:.6f}, "
                              f"LR: {self.current_learning_rate:.2e}, LR Reductions: {self.learning_rate_reductions}")
                
                # Checkpoint saving
                if (epoch + 1) % self.training_config['checkpoint_interval'] == 0:
                    actual_epoch = start_epoch + epoch + 1
                    checkpoint_filename = f"checkpoint_epoch_{actual_epoch}.npz"
                    if hasattr(self, 'persistence') and self.persistence is not None:
                        current_time = time.time()
                        total_training_time = current_time - self.training_start_time
                        checkpoint_metadata = {
                            'epoch': actual_epoch,
                            'best_val_loss': best_val_loss,
                            'training_loss': epoch_loss,
                            'validation_loss': val_loss,
                            'training_time': total_training_time,
                            'epoch_duration': epoch_duration,
                            'average_epoch_time': total_training_time / (epoch + 1),
                            'patience_counter': patience_counter,
                            'early_stopping_patience': self.training_config['early_stopping_patience'],
                            'epochs_without_improvement': patience_counter,
                            'checkpoint_type': 'training',
                            'timestamp': current_time,
                            'training_examples': len(train_samples),
                            'validation_examples': len(val_samples)
                        }
                        success = self.persistence.save_network_weights(checkpoint_filename, checkpoint_metadata)
                        if success:
                            logger.info(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_filename}")
                        else:
                            logger.warning(f"Failed to save checkpoint at epoch {epoch + 1}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model checkpoint
                    if hasattr(self, 'persistence') and self.persistence is not None:
                        current_time = time.time()
                        total_training_time = current_time - self.training_start_time
                        best_model_metadata = {
                            'epoch': start_epoch + epoch + 1,
                            'best_val_loss': best_val_loss,
                            'training_loss': epoch_loss,
                            'validation_loss': val_loss,
                            'training_time': total_training_time,
                            'epoch_duration': epoch_duration,
                            'average_epoch_time': total_training_time / (epoch + 1),
                            'patience_counter': 0,  # Reset since this is a new best
                            'early_stopping_patience': self.training_config['early_stopping_patience'],
                            'epochs_without_improvement': 0,
                            'checkpoint_type': 'best_model',
                            'timestamp': current_time,
                            'training_examples': len(train_samples),
                            'validation_examples': len(val_samples)
                        }
                        self.persistence.save_network_weights("best_model.npz", best_model_metadata)
                        logger.info(f"New best model saved at epoch {start_epoch + epoch + 1} with val loss: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.training_config['early_stopping_patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Restore original loss function
            self.current_network.loss_function_and_grad = original_loss_func
            
            training_time = time.time() - self.training_start_time
            logger.info(f"Pretraining completed! Final validation loss: {best_val_loss:.6f}")
            logger.info(f"Training time: {training_time/60:.1f} minutes")
            
            # After training completes, copy best model to final_trained_model.npz
            if hasattr(self, 'persistence') and self.persistence is not None:
                import shutil
                best_model_path = os.path.join(constants.PATH, "best_model.npz")
                final_model_path = os.path.join(constants.PATH, "final_trained_model.npz")

                if os.path.exists(best_model_path):
                    # Backup existing final model if it exists
                    if os.path.exists(final_model_path):
                        backup_path = os.path.join(constants.PATH, "final_trained_model.backup.npz")
                        shutil.copy2(final_model_path, backup_path)
                        logger.info(f"Backed up existing final model to {backup_path}")

                    # Copy best model to final
                    shutil.copy2(best_model_path, final_model_path)
                    logger.info(f"Copied best model (epoch with val loss {best_val_loss:.6f}) to final_trained_model.npz")
                else:
                    logger.warning("Best model file not found - final_trained_model.npz not updated")

            return {
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'best_val_loss': best_val_loss,
                'best_model_epoch': best_model_metadata.get('epoch', 0) if 'best_model_metadata' in locals() else 0,
                'epochs_completed': epoch + 1,
                'training_time': training_time,
                'training_examples': len(train_samples)
            }

        except Exception as e:
            logger.error(f"Pretraining failed: {str(e)}")
            raise constants.physics_parameter_error(f"Pretraining failed: {str(e)}")
    
    def train_physics_based(self, params: List[Union[int, float]], iterations: int = 50,
                           loss_weights: Optional[List[float]] = None) -> dict:
        """
        Train network using physics-based loss for specific TBG parameters.
        
        Args:
            params (List[Union[int, float]]): TBG parameters
            iterations (int): Number of optimization iterations
            loss_weights (List[float], optional): Weights for loss components
            
        Returns:
            dict: Training results
            
        Raises:
            constants.physics_parameter_error: If network builder not available or training fails
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network set")
        if self.current_network_builder is None:
            raise constants.physics_parameter_error("Network builder required for physics training")
        
        loss_weights = loss_weights or default_loss_weights
        
        try:
            logger.info("Starting physics-based training...")
            
            # Set parameters using builder
            validated_params = self.current_network_builder.set_network_parameters(params)
            
            # Set physics-based loss function
            # Wrap loss function so backward() doesn't need to pass weights
            def wrapped_physics_loss(output, weights=None):
                # Ignore weights parameter from backward(), use proper physics weights
                return self._physics_loss_function(output, loss_weights)

            self.current_network.loss_function_and_grad = wrapped_physics_loss
            
            # Get initial loss
            initial_output = self.current_network.compute()
            initial_loss, _ = self._physics_loss_function(initial_output, loss_weights)
            
            logger.info(f"Initial physics loss: {initial_loss:.6f}")
            
            # Training loop
            physics_losses = []
            for i in range(iterations):
                self.current_network.update_weights()
                
                if i % 10 == 0 or i == iterations - 1:
                    output = self.current_network.compute()
                    physics_loss, _ = self._physics_loss_function(output, loss_weights)
                    physics_losses.append(physics_loss)
                    logger.info(f"Iteration {i}: Physics loss = {physics_loss:.6f}")
            
            final_output = self.current_network.compute()
            final_loss, _ = self._physics_loss_function(final_output, loss_weights)
            
            improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
            logger.info(f"Physics training complete. Improvement: {improvement:.1f}%")
            
            return {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement_percent': improvement,
                'physics_losses': physics_losses,
                'iterations': iterations,
                'final_prediction': final_output
            }
            
        except Exception as e:
            logger.error(f"Physics-based training failed: {str(e)}")
            raise constants.physics_parameter_error(f"Physics training failed: {str(e)}")
    
    def _train_epoch_grouped(self, training_samples: List[Dict[str, Any]], batch_size: int) -> float:
        """
        Train for one epoch using grouped data with minimum distance loss.

        Args:
            training_samples (List[Dict[str, Any]]): List of parameter sets with their Dirac points
            batch_size (int): Size of mini-batches

        Returns:
            float: Average epoch loss
        """
        epoch_loss = 0.0
        num_samples = 0

        # CRITICAL: Ensure minimum distance loss function is set
        # Wrap loss function so backward() doesn't need to pass weights
        def wrapped_min_dist_loss(output, weights=None):
            # Ignore weights parameter from backward(), use proper supervised weights
            loss, gradients, _, _ = self._minimum_distance_loss_function(output, constants.DEFAULT_SUPERVISED_LOSS_WEIGHTS)
            return loss, gradients

        self.current_network.loss_function_and_grad = wrapped_min_dist_loss

        # Reset ν clamping counters at start of each epoch
        self.nu_clamp_count = 0
        self.batch_sample_count = 0

        # Log initial network state before any training
        for neuron in self.current_network.layers[0].neurons:
            neuron.output = 0.0
        init_output = self.current_network.compute()
        output_layer = self.current_network.layers[-1]
        k_x1_init_bias = output_layer.neurons[0].bias if hasattr(output_layer.neurons[0], 'bias') else 0.0
        k_y1_init_bias = output_layer.neurons[1].bias if hasattr(output_layer.neurons[1], 'bias') else 0.0
        k_x2_init_bias = output_layer.neurons[3].bias if hasattr(output_layer.neurons[3], 'bias') else 0.0
        k_y2_init_bias = output_layer.neurons[4].bias if hasattr(output_layer.neurons[4], 'bias') else 0.0
        logger.info(f"INITIAL STATE (before training): Pred1 = [{init_output[0]:.4f}, {init_output[1]:.4f}, {init_output[2]:.4f}], "
                   f"Pred2 = [{init_output[3]:.4f}, {init_output[4]:.4f}, {init_output[5]:.4f}], "
                   f"Biases = [k_x1:{k_x1_init_bias:.4f}, k_y1:{k_y1_init_bias:.4f}, k_x2:{k_x2_init_bias:.4f}, k_y2:{k_y2_init_bias:.4f}]")

        # Track gradient statistics for monitoring explosion/vanishing
        gradient_k_x = []
        gradient_k_y = []
        gradient_nu = []
        gradient_total = []

        for i in range(0, len(training_samples), batch_size):
            batch_samples = training_samples[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(training_samples) - 1) // batch_size + 1

            batch_loss = 0.0

            # Standard training: forward pass, compute gradients, update weights
            for sample in batch_samples:
                # Use TRANSFORMED parameters (1/a, 1/b, ...) for network input
                # This is critical for numerical stability and learning
                params_transformed = sample['parameters_transformed']
                dirac_points = sample['dirac_points']

                self._set_network_inputs(params_transformed)
                self._set_target_outputs(dirac_points)

                # Forward pass
                output = self.current_network.compute()

                # Compute loss and gradients (with repulsion built-in)
                loss, gradients, closest_target1, closest_target2 = self._minimum_distance_loss_function(
                    output, constants.DEFAULT_SUPERVISED_LOSS_WEIGHTS
                )

                # Track gradient magnitudes for all 6 outputs
                gradient_k_x.append(abs(gradients[0]))  # k_x1
                gradient_k_y.append(abs(gradients[1]))  # k_y1
                gradient_nu.append(abs(gradients[2]))   # nu1
                # Also track second prediction gradients
                gradient_k_x.append(abs(gradients[3]))  # k_x2
                gradient_k_y.append(abs(gradients[4]))  # k_y2
                gradient_nu.append(abs(gradients[5]))   # nu2
                gradient_total.append(np.linalg.norm(gradients))

                # Apply gradients via custom backprop
                self.current_network.backward_with_custom_gradients(gradients)

                # Update weights with computed gradients
                for layer_obj in self.current_network.layers:
                    layer_obj.update_weights()

                batch_loss += loss
                num_samples += 1

            # Print batch progress: every batch for first 20, then every 10 batches
            should_log = (batch_num <= 20) or (batch_num % 10 == 0) or (batch_num == total_batches)
            if should_log:
                avg_batch_loss = batch_loss / len(batch_samples)

                # Sample current network output without changing inputs
                network_outputs = "N/A"
                hidden_info = ""
                repulsion_info = ""
                try:
                    # Just compute with whatever inputs are currently set
                    raw_output = self.current_network.compute()
                    if len(raw_output) >= 6:
                        # Show both predictions
                        network_outputs = f"Pred1=[{raw_output[0]:.4f}, {raw_output[1]:.4f}, {raw_output[2]:.4f}], Pred2=[{raw_output[3]:.4f}, {raw_output[4]:.4f}, {raw_output[5]:.4f}]"

                        # Calculate k-space separation for monitoring repulsion
                        k_sep = np.sqrt((raw_output[0] - raw_output[3])**2 + (raw_output[1] - raw_output[4])**2)
                        repulsion_info = f", k-sep={k_sep:.4f}"

                        # Get hidden layer statistics for monitoring saturation
                        output_layer = self.current_network.layers[-1]
                        hidden_layer = self.current_network.layers[-2]  # Last hidden layer
                        hidden_outputs = np.array([n.output for n in hidden_layer.neurons])
                        hidden_mag = np.mean(np.abs(hidden_outputs))
                        hidden_max = np.max(np.abs(hidden_outputs))
                        hidden_info = f", Hidden: {hidden_mag:.2f}+/-{hidden_max:.2f}"

                        # Always show bias values for first 20 batches to diagnose collapse
                        if (batch_num <= 20 or logger.isEnabledFor(logging.DEBUG)) and len(output_layer.neurons) >= 2:
                            k_x_neuron = output_layer.neurons[0]
                            k_y_neuron = output_layer.neurons[1]
                            k_x_pre = k_x_neuron.sum
                            k_y_pre = k_y_neuron.sum
                            k_x_weights = np.array([w for _, w in k_x_neuron.inputs])
                            k_y_weights = np.array([w for _, w in k_y_neuron.inputs])
                            k_x_weights_mag = np.mean(np.abs(k_x_weights))
                            k_y_weights_mag = np.mean(np.abs(k_y_weights))
                            k_x_bias = k_x_neuron.bias if hasattr(k_x_neuron, 'bias') else 0.0
                            k_y_bias = k_y_neuron.bias if hasattr(k_y_neuron, 'bias') else 0.0

                            # Use info level for first 20 batches, debug otherwise
                            log_func = logger.info if batch_num <= 20 else logger.debug
                            log_func(f"   Pre-activation: [{k_x_pre:.4f}, {k_y_pre:.4f}], "
                                   f"Weights: [{k_x_weights_mag:.4f}, {k_y_weights_mag:.4f}], "
                                   f"Biases: [{k_x_bias:.4f}, {k_y_bias:.4f}]")
                    else:
                        network_outputs = f"{raw_output}"
                except Exception as e:
                    network_outputs = f"Error: {str(e)[:20]}"

                # Calculate clamping statistics
                clamp_percentage = (self.nu_clamp_count / self.batch_sample_count * 100) if self.batch_sample_count > 0 else 0

                # Calculate gradient statistics (mean and max for recent samples in this batch)
                recent_grad_k_x = gradient_k_x[-len(batch_samples):] if gradient_k_x else [0]
                recent_grad_k_y = gradient_k_y[-len(batch_samples):] if gradient_k_y else [0]
                recent_grad_nu = gradient_nu[-len(batch_samples):] if gradient_nu else [0]
                recent_grad_total = gradient_total[-len(batch_samples):] if gradient_total else [0]

                grad_stats = (f"Grads: k_x={np.mean(recent_grad_k_x):.2e}+/-{np.max(recent_grad_k_x):.2e}, "
                             f"k_y={np.mean(recent_grad_k_y):.2e}+/-{np.max(recent_grad_k_y):.2e}, "
                             f"nu={np.mean(recent_grad_nu):.2e}+/-{np.max(recent_grad_nu):.2e}, "
                             f"total={np.mean(recent_grad_total):.2e}+/-{np.max(recent_grad_total):.2e}")

                logger.info(f"   Batch {batch_num}/{total_batches}: Avg Loss = {avg_batch_loss:.4f}, {network_outputs}{repulsion_info}{hidden_info}, nu Clamped: {self.nu_clamp_count}/{self.batch_sample_count} ({clamp_percentage:.1f}%)")
                logger.info(f"   {grad_stats}")

                # Reset counters after logging every 100 batches
                if batch_num % 100 == 0:
                    self.nu_clamp_count = 0
                    self.batch_sample_count = 0

            epoch_loss += batch_loss

        return epoch_loss / num_samples

    def _evaluate_validation_loss_grouped(self, validation_samples: List[Dict[str, Any]]) -> float:
        """
        Evaluate loss on validation set using grouped data.

        Args:
            validation_samples (List[Dict[str, Any]]): List of parameter sets with their Dirac points

        Returns:
            float: Average validation loss
        """
        # Clear any previous state to avoid contamination
        self._current_targets = None
        total_loss = 0.0

        # Shuffle validation set each time to ensure different evaluation order
        val_indices = np.random.permutation(len(validation_samples))

        for i in val_indices:
            sample = validation_samples[i]
            # Use TRANSFORMED parameters (1/a, 1/b, ...) for network input
            params_transformed = sample['parameters_transformed']
            dirac_points = sample['dirac_points']

            self._set_network_inputs(params_transformed)
            self._set_target_outputs(dirac_points)

            # Compute fresh network output (no caching)
            output = self.current_network.compute()
            loss, _, _, _ = self._minimum_distance_loss_function(output, constants.DEFAULT_SUPERVISED_LOSS_WEIGHTS)
            total_loss += loss

        # Clear targets after validation to prevent interference
        self._current_targets = None
        return total_loss / len(validation_samples)

    def _set_network_inputs(self, input_values: np.ndarray) -> None:
        """Set input layer values for training."""
        if len(input_values) != len(self.current_network.layers[0].neurons):
            raise constants.matrix_operation_error(
                f"Input size mismatch: got {len(input_values)}, expected {len(self.current_network.layers[0].neurons)}"
            )
        
        for i, neuron in enumerate(self.current_network.layers[0].neurons):
            if isinstance(neuron, dummy_neuron):
                neuron.output = input_values[i]
    
    def _set_target_output(self, target: np.ndarray) -> None:
        """Store target output for loss computation."""
        self._current_target = target

    def _set_target_outputs(self, targets: List[np.ndarray]) -> None:
        """
        Store multiple target outputs for minimum distance loss computation.

        Args:
            targets: List of target arrays, each [k_x, k_y, velocity]
        """
        self._current_targets = targets
    
    def _mse_loss_function(self, output: List[float], weights: List[float]) -> Tuple[float, np.ndarray]:
        """
        Mean Squared Error loss function for data-based pretraining.

        Args:
            output: Network output [k_x, k_y, velocity]
            weights: Dummy weights (not used for MSE)

        Returns:
            Tuple[float, np.ndarray]: (loss, gradients)
        """
        if not hasattr(self, '_current_target') or self._current_target is None:
            return 0.0, np.zeros(3)

        # Clamp ν (third output) to valid range to prevent physics violations
        output_clamped = output.copy()
        output_clamped[2] = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, output[2]))

        # Track clamping statistics
        if abs(output[2] - output_clamped[2]) > 1e-6:
            self.nu_clamp_count += 1
        self.batch_sample_count += 1

        output_array = np.array(output_clamped)
        target_array = np.array(self._current_target)

        # Standard MSE loss and gradients (no artificial scaling)
        diff = output_array - target_array
        loss = 0.5 * np.sum(diff ** 2)
        gradients = diff  # Pure MSE gradients without scaling tricks

        return loss, gradients

    def _minimum_distance_loss_function(self, output: List[float], weights: List[float]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-point minimum distance loss with Coulomb repulsion.

        Predicts TWO Dirac points with repulsion between them to prevent collapse.
        Each prediction finds its closest valid target. Repulsion forces diversity.

        Args:
            output: Network output [k_x1, k_y1, nu1, k_x2, k_y2, nu2]
            weights: Loss weights [k_x_weight, k_y_weight, nu_weight] applied to EACH prediction
                     Default: [0.6, 0.6, 0.4] to emphasize k-points

        Returns:
            Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
                (total_loss, gradient_vector, closest_target1, closest_target2)
                total_loss: loss1 + loss2 + repulsion
                gradient_vector: gradients for all 6 outputs
                closest_target1: closest target for first prediction
                closest_target2: closest target for second prediction
        """
        if not hasattr(self, '_current_targets') or self._current_targets is None:
            return 0.0, np.zeros(6), np.zeros(3), np.zeros(3)

        # Extract two predictions from output
        pred1 = np.array([output[0], output[1], output[2]])  # [k_x1, k_y1, nu1]
        pred2 = np.array([output[3], output[4], output[5]])  # [k_x2, k_y2, nu2]

        # Clamp nu values to valid range
        pred1[2] = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, pred1[2]))
        pred2[2] = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, pred2[2]))

        # Track clamping statistics
        if abs(output[2] - pred1[2]) > 1e-6:
            self.nu_clamp_count += 1
        if abs(output[5] - pred2[2]) > 1e-6:
            self.nu_clamp_count += 1
        self.batch_sample_count += 1

        # Use default weights if not provided
        if weights is None or len(weights) != 3:
            loss_weights = np.array(constants.DEFAULT_SUPERVISED_LOSS_WEIGHTS)
        else:
            loss_weights = np.array(weights)

        # Find closest target for prediction 1
        min_loss1 = float('inf')
        closest_idx1 = 0
        for idx, target in enumerate(self._current_targets):
            # Convert target from [k_x, k_y, velocity] to [k_x, k_y, nu]
            # NN outputs nu, but targets contain velocity, so we must convert
            target_k_x, target_k_y, target_velocity = target[0], target[1], target[2]
            target_nu = 1.0 / (1.0 + abs(target_velocity))  # Convert velocity -> nu
            target_array = np.array([target_k_x, target_k_y, target_nu])

            diff = pred1 - target_array
            weighted_loss = 0.5 * np.sum(loss_weights * diff ** 2)
            if weighted_loss < min_loss1:
                min_loss1 = weighted_loss
                closest_idx1 = idx

        # Find closest target for prediction 2
        min_loss2 = float('inf')
        closest_idx2 = 0
        for idx, target in enumerate(self._current_targets):
            # Convert target from [k_x, k_y, velocity] to [k_x, k_y, nu]
            target_k_x, target_k_y, target_velocity = target[0], target[1], target[2]
            target_nu = 1.0 / (1.0 + abs(target_velocity))  # Convert velocity -> nu
            target_array = np.array([target_k_x, target_k_y, target_nu])

            diff = pred2 - target_array
            weighted_loss = 0.5 * np.sum(loss_weights * diff ** 2)
            if weighted_loss < min_loss2:
                min_loss2 = weighted_loss
                closest_idx2 = idx

        # Compute Coulomb repulsion in k-space only (not nu)
        k_diff = np.array([pred1[0] - pred2[0], pred1[1] - pred2[1]])
        k_distance = np.sqrt(k_diff[0]**2 + k_diff[1]**2)
        repulsion_energy = constants.REPULSION_WEIGHT / (k_distance + constants.REPULSION_EPSILON)

        # Total loss
        total_loss = min_loss1 + min_loss2 + repulsion_energy

        # Gradients for prediction 1
        # Convert closest target from [k_x, k_y, velocity] to [k_x, k_y, nu]
        closest_target1_raw = self._current_targets[closest_idx1]
        target1_nu = 1.0 / (1.0 + abs(closest_target1_raw[2]))
        closest_target1 = np.array([closest_target1_raw[0], closest_target1_raw[1], target1_nu])
        diff1 = pred1 - closest_target1
        grad1 = loss_weights * diff1

        # Gradients for prediction 2
        # Convert closest target from [k_x, k_y, velocity] to [k_x, k_y, nu]
        closest_target2_raw = self._current_targets[closest_idx2]
        target2_nu = 1.0 / (1.0 + abs(closest_target2_raw[2]))
        closest_target2 = np.array([closest_target2_raw[0], closest_target2_raw[1], target2_nu])
        diff2 = pred2 - closest_target2
        grad2 = loss_weights * diff2

        # Repulsion gradients (only for k-space, not nu)
        # d/d(pred1) [W / (|pred1 - pred2| + eps)] = -W * (pred1 - pred2) / (|pred1 - pred2| + eps)^2 / |pred1 - pred2|
        repulsion_grad_magnitude = -constants.REPULSION_WEIGHT / ((k_distance + constants.REPULSION_EPSILON)**2 * (k_distance + 1e-10))
        repulsion_grad_k = repulsion_grad_magnitude * k_diff

        # Add repulsion gradients to k-components only
        grad1[0] += repulsion_grad_k[0]  # k_x1
        grad1[1] += repulsion_grad_k[1]  # k_y1
        grad2[0] += -repulsion_grad_k[0]  # k_x2 (opposite direction)
        grad2[1] += -repulsion_grad_k[1]  # k_y2 (opposite direction)

        # Combine gradients into single vector
        gradients = np.concatenate([grad1, grad2])

        return total_loss, gradients, closest_target1, closest_target2
    
    def _physics_loss_function(self, output: List[float], weights: List[float]) -> Tuple[float, np.ndarray]:
        """
        Physics-based Dirac point loss function.
        
        Args:
            output: Network output [k_x, k_y, velocity]
            weights: Weights for loss components [gap, R², isotropy]
            
        Returns:
            Tuple[float, np.ndarray]: (loss, gradients)
        """
        if self.current_network_builder is None or self.current_network_builder.current_periodic_graph is None:
            raise constants.physics_parameter_error("No physics graphs available for loss computation")
        
        if len(weights) != 3 or len(output) != 3:
            raise ValueError("Must provide 3 weights and 3 outputs")

        # Clamp ν (third output) to valid range to prevent physics violations
        output_clamped = list(output)
        output_clamped[2] = max(constants.NU_CLAMP_MIN, min(constants.NU_CLAMP_MAX, output[2]))

        # Track clamping statistics
        if abs(output[2] - output_clamped[2]) > 1e-6:
            self.nu_clamp_count += 1
        self.batch_sample_count += 1

        k_point = output_clamped[:2]
        velocity_pred = output_clamped[2]
        
        # Compute Dirac point metrics and their gradients
        dirac_analyzer = Dirac_analysis(self.current_network_builder.current_periodic_graph)
        metrics, metrics_der_n_1, metrics_der_n_2, velocity_calc, velocity_der_n_1, velocity_der_n_2 = dirac_analyzer.check_Dirac_point(k_point, 1)
        
        # Weighted combination of loss components
        dirac_loss = np.sum(np.array(metrics) * np.array(weights))
        velocity_loss = velocity_calc - velocity_pred
        loss_function = dirac_loss**2 + velocity_loss**2
        
        # Compute gradients
        loss_function_der_n_1 = 2*dirac_loss*np.sum(np.array(metrics_der_n_1)*np.array(weights)) + 2*velocity_loss*velocity_der_n_1
        loss_function_der_n_2 = 2*dirac_loss*np.sum(np.array(metrics_der_n_2)*np.array(weights)) + 2*velocity_loss*velocity_der_n_2
        loss_function_der_vel = -2*velocity_loss
        
        return loss_function, np.array([loss_function_der_n_1, loss_function_der_n_2, loss_function_der_vel])
    
    def _detect_loss_explosion(self, current_loss: float) -> bool:
        """
        Detect if loss has exploded based on recent history.

        Args:
            current_loss (float): Current epoch loss

        Returns:
            bool: True if loss explosion detected
        """
        # Add current loss to history
        self.loss_history.append(current_loss)

        # Keep only recent history
        if len(self.loss_history) > self.training_config['loss_history_window']:
            self.loss_history.pop(0)

        # Check for loss explosion
        if current_loss > self.training_config['loss_explosion_threshold']:
            logger.warning(f"Loss explosion detected: {current_loss:.6f} > {self.training_config['loss_explosion_threshold']}")
            return True

        # Check for sudden spike (current loss much larger than recent average)
        if len(self.loss_history) >= 3:
            recent_avg = np.mean(self.loss_history[:-1])  # Average excluding current loss
            if current_loss > 5.0 * recent_avg and recent_avg > 0:
                logger.warning(f"Sudden loss spike detected: {current_loss:.6f} vs recent avg {recent_avg:.6f}")
                return True

        return False

    def _reduce_learning_rate(self) -> bool:
        """
        Reduce learning rate to recover from loss explosion.

        Returns:
            bool: True if learning rate was reduced, False if already at minimum
        """
        new_lr = self.current_learning_rate * self.training_config['learning_rate_reduction_factor']

        if new_lr < self.training_config['min_learning_rate']:
            logger.warning(f"Learning rate already at minimum: {self.current_learning_rate:.8f}")
            return False

        self.current_learning_rate = new_lr
        self.learning_rate_reductions += 1

        # Update learning rate in all ADAM optimizers
        if self.current_network is not None:
            self._update_network_learning_rate(new_lr)

        logger.info(f"Learning rate reduced to {new_lr:.8f} (reduction #{self.learning_rate_reductions})")
        return True

    def _update_network_learning_rate(self, new_lr: float) -> None:
        """
        Update learning rate in all ADAM optimizers in the network.

        Args:
            new_lr (float): New learning rate
        """
        for layer in self.current_network.layers:
            if not layer.Dummy:  # Skip input layers
                for neuron in layer.neurons:
                    # Update learning rate in all ADAM optimizers for this neuron
                    if hasattr(neuron, 'Adam_corrector'):
                        for adam_opt in neuron.Adam_corrector:
                            adam_opt.alpha = new_lr
                    # Update bias ADAM optimizer if present
                    if hasattr(neuron, 'bias_adam') and neuron.bias_adam is not None:
                        neuron.bias_adam.alpha = new_lr
    

    def get_training_statistics(self) -> dict:
        """Get training statistics summary."""
        return {
            'stats_object': self.stats,
            'training_start_time': self.training_start_time,
            'has_active_network': self.current_network is not None,
            'current_learning_rate': self.current_learning_rate,
            'initial_learning_rate': self.training_config['learning_rate'],
            'learning_rate_reductions': self.learning_rate_reductions,
            'loss_history_length': len(self.loss_history),
            'gradient_clip_value': self.training_config['gradient_clip_value']
        }
    
    def reset(self) -> None:
        """Reset trainer state."""
        self.current_network = None
        self.current_network_builder = None
        self._current_target = None
        self.stats.cleanup()
        self.training_start_time = None

        # Reset adaptive learning rate tracking
        self.current_learning_rate = self.training_config['learning_rate']
        self.loss_history = []
        self.learning_rate_reductions = 0

        # Reset ν clamping statistics
        self.nu_clamp_count = 0
        self.batch_sample_count = 0

        logger.info("dirac_network_trainer state reset")