"""
Refactored Dirac point neural network using modular architecture.

This module provides a clean interface for Dirac point prediction using
specialized classes for network building, training, benchmarking, and persistence.
Follows project coding standards and integrates existing utilities.

Classes:
    nn_dirac_point: Main orchestrator class for Dirac point neural networks
"""

import constants
from constants import np, os, time, glob, logging
from constants import List, Tuple, Optional, Union, Any, Dict
from neural_network_base import neural_network, dummy_neuron
from dirac_network_builder import dirac_network_builder
from dirac_network_trainer import dirac_network_trainer
from dirac_network_benchmark import dirac_network_benchmark
from dirac_network_persistence import dirac_network_persistence
from stats import statistics
from TBG import tbg, Dirac_analysis
from utils import compute_twist_constants
from simulation_data_loader import simulation_data_analyzer

# Configure global logging
logger = logging.getLogger(__name__)

# Default configuration for the orchestrator
default_orchestrator_config = {
    'network_config': None,  # Use builder defaults
    'training_config': None,  # Use trainer defaults
    'benchmark_config': None,  # Use benchmark defaults
    'persistence_config': None,  # Use persistence defaults
    'auto_save': True,
    'default_model_name': 'dirac_nn_model.npz'
}


class nn_dirac_point:
    """
    Main orchestrator class for Dirac point neural networks using modular architecture.
    
    This class coordinates between specialized modules for network building, training,
    benchmarking, and persistence to provide a clean interface for Dirac point prediction.
    Follows project coding standards and integrates existing utilities.
    
    The network takes TBG parameters as input and predicts Dirac point coordinates:
    - Input: [a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight]
    - Output: [k_x, k_y, velocity]
    
    Attributes:
        network (neural_network): The underlying neural network
        builder (dirac_network_builder): Network construction handler
        trainer (dirac_network_trainer): Training functionality handler
        benchmark (dirac_network_benchmark): Performance benchmarking handler
        persistence (dirac_network_persistence): Save/load functionality handler
        config (dict): Orchestrator configuration
    """
    def __init__(self, config: dict = None) -> None:
        """
        Initialize the Dirac point neural network orchestrator.
        
        Args:
            config (dict, optional): Configuration parameters. Uses defaults if None.
        """
        # Set configuration
        self.config = {**default_orchestrator_config, **(config or {})}
        
        # Initialize specialized modules
        self.builder = dirac_network_builder(self.config.get('network_config'))
        self.trainer = dirac_network_trainer(self.config.get('training_config'))
        self.benchmark = dirac_network_benchmark(self.config.get('benchmark_config'))
        self.persistence = dirac_network_persistence(
            base_path=constants.PATH, 
            persistence_config=self.config.get('persistence_config')
        )
        
        # Build the network using the builder
        self.network = self.builder.build_network()
        
        # Set the network in all modules
        self.trainer.set_network(self.network, self.builder)
        self.benchmark.set_network(self.network, self.builder)
        self.persistence.set_network(self.network)
        
        # Connect persistence to trainer for checkpoint saving
        self.trainer.set_persistence(self.persistence)
        
        # Initialize statistics tracking
        self.stats = statistics()
        
        logger.info("Dirac point neural network orchestrator initialized")

    def set_parameters(self, params: List[Union[Tuple[int, int], float]]) -> None:
        """
        Set TBG parameters for prediction or physics-based optimization.
        
        Args:
            params: List containing [a, b, interlayer_threshold, intralayer_threshold, 
                   inter_weight, intra_weight]
        """
        # Delegate to builder for parameter setup and network input setting
        self.builder.set_network_parameters(params)

    def predict(self, params: List[Union[int, float]]) -> Tuple[float, float, float]:
        """
        Predict Dirac point for given TBG parameters with output transformations.
        
        Args:
            params: List containing [a, b, interlayer_threshold, intralayer_threshold, 
                   inter_weight, intra_weight]
                   
        Returns:
            Tuple of (k_x, k_y, velocity) with ν transformed back to velocity
        """
        self.set_parameters(params)
        output = self.network.compute()
        
        # Convert ν = 1/(1+v) back to velocity: v = (1-ν)/ν
        k_x, k_y, nu = output[0], output[1], output[2]
        
        # Handle edge cases for ν transformation
        if nu <= 0.0001:  # Avoid division by zero, corresponds to very large velocity
            velocity = 10000.0  # Cap at large value
        else:
            velocity = (1.0 - nu) / nu
            
        return k_x, k_y, velocity
    
    def predict_timed(self, params: List[Union[int, float]]) -> Tuple[Tuple[float, float, float], float]:
        """
        Predict Dirac point with timing measurement.
        
        Args:
            params: TBG parameters
            
        Returns:
            Tuple of ((k_x, k_y, velocity), prediction_time)
        """
        # Delegate to benchmark module
        return self.benchmark.benchmark_prediction_time(params, iterations=1)
    
    def physics_computation_timed(self, params: List[Union[int, float]], k_point: List[float]) -> Tuple[List[float], float]:
        """
        Perform physics-based Dirac point computation with timing.
        
        Args:
            params: TBG parameters
            k_point: Initial k-point guess
            
        Returns:
            Tuple of (metrics, computation_time)
        """
        # Delegate to benchmark module
        return self.benchmark.benchmark_physics_computation(params, k_point, iterations=1)
    
    def benchmark_acceleration_factor(self, test_params_list: List[List[Union[int, float]]], 
                                    num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark neural network vs physics computation speeds and calculate acceleration factor.
        
        Args:
            test_params_list: List of parameter sets to test
            num_iterations: Number of iterations per parameter set
            
        Returns:
            Dictionary with benchmark results and acceleration factor
        """
        # Delegate to benchmark module
        return self.benchmark.calculate_acceleration_factor(test_params_list, num_iterations)
    
    def log_training_statistics(self) -> None:
        """
        Log comprehensive neural network training statistics.
        """
        # Delegate to benchmark module for performance statistics
        performance_report = self.benchmark.generate_performance_report()
        
        # Also use main statistics object
        self.stats.log_statistics()
        
        # Log trainer statistics if available
        trainer_stats = self.trainer.get_training_statistics()
        if trainer_stats:
            logger.info("Training statistics available")
            for key, value in trainer_stats.items():
                logger.info(f"{key}: {value}")
        
        return performance_report
    
    def save_training_statistics(self, filename: str = None) -> None:
        """
        Save neural network training statistics to files.
        
        Args:
            filename: Base filename for statistics files
        """
        # Delegate to benchmark module for performance statistics
        self.benchmark.save_benchmark_results(filename)
        
        # Also save main statistics using utility
        stats_filename = filename or "nn_training_statistics.csv"
        self.persistence.save_statistics_using_utility(self.stats, stats_filename)

    def loss_function_and_derivative(self, output: List[float], 
                                   weights: List[float]) -> Tuple[float, np.ndarray]:
        """
        Compute the Dirac point loss function and its gradients.
        Delegates to trainer module for physics-based loss computation.
        """
        # Delegate to trainer module which handles physics-based loss
        return self.trainer._physics_loss_function(output, weights)
    
    def pretrain_from_data(self, data_folder: str = None, epochs: int = 200, 
                          batch_size: int = 128, validation_split: float = 0.1) -> None:
        """
        Pretrain the neural network using historical Dirac point data.
        
        This method loads training data from CSV files and trains the network to predict
        Dirac point coordinates and velocities from TBG parameters before fine-tuning
        with the physics-based optimization. Learning rate is managed by trainer configuration.
        
        Args:
            data_folder (str, optional): Path to folder containing training data. 
                Defaults to None (uses default data folder).
            epochs (int): Number of training epochs. Defaults to 200.
            batch_size (int): Batch size for training. Defaults to 128.
            validation_split (float): Fraction of data to use for validation. Defaults to 0.1.
        """
        # Delegate to trainer module for pretraining
        return self.trainer.pretrain_from_data(data_folder, epochs, batch_size, validation_split)
    
    
    def train_hybrid(self, target_params: List[Union[int, float]], data_folder: str = None, 
                    pretrain_epochs: int = 200, physics_iterations: int = 100) -> Tuple[float, float]:
        """
        Complete hybrid training: pretrain on data, then optimize with physics for specific parameters.
        
        This method first pretrains the network on historical data to learn general
        patterns, then fine-tunes using the physics-based loss function for a specific parameter set.
        Learning rate is managed by trainer configuration.
        
        Args:
            target_params: List of [a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight]
            data_folder (str, optional): Path to training data folder
            pretrain_epochs (int): Epochs for pretraining phase. Defaults to 200.
            physics_iterations (int): Iterations for physics optimization. Defaults to 100.
            
        Returns:
            Tuple[float, float]: (final_pretrain_loss, final_physics_loss)
        """
        # First phase: pretrain on data if data folder provided
        if data_folder is not None:
            self.pretrain_from_data(data_folder, pretrain_epochs)
        
        # Second phase: physics-based training for specific parameters
        return self.trainer.train_physics_based(target_params, physics_iterations)
    
    def train_general(self, data_folder: str = None, epochs: int = 300, learning_rate: float = 0.001,
                     batch_size: int = 64, validation_split: float = 0.15,
                     save_checkpoints: bool = True, checkpoint_interval: int = 50,
                     early_stopping_patience: int = 100, start_epoch: int = 0,
                     resume_patience_counter: int = 0) -> Dict[str, Any]:
        """
        General-purpose training method focused on comprehensive statistics and performance analysis.
        
        This method provides a complete training workflow with detailed monitoring, statistics
        collection, and performance benchmarking. It trains on all available data without
        targeting specific physics parameters.
        
        Args:
            data_folder (str, optional): Path to training data folder
            epochs (int): Maximum number of training epochs. Defaults to 300.
            learning_rate (float): Learning rate for training. Defaults to 0.001.
            batch_size (int): Batch size for training. Defaults to 64.
            validation_split (float): Fraction of data for validation. Defaults to 0.15.
            save_checkpoints (bool): Whether to save model checkpoints. Defaults to True.
            checkpoint_interval (int): Epochs between checkpoints. Defaults to 50.
            early_stopping_patience (int): Epochs to wait before early stopping. Defaults to 100.
            start_epoch (int): Epoch number to start/resume from. Defaults to 0.
            resume_patience_counter (int): Resume early stopping patience counter. Defaults to 0.

        Returns:
            Dict[str, Any]: Comprehensive training results and statistics
        """
        # Perform comprehensive training using trainer's method (handles data loading internally)
        training_results = self.trainer.pretrain_from_data(
            data_folder=data_folder,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            start_epoch=start_epoch,
            resume_patience_counter=resume_patience_counter
        )
        
        # Auto-save if configured
        if self.config.get('auto_save', False):
            self.save_weights(self.config.get('default_model_name', 'trained_model.npz'))
        
        return {
            'training_completed': True,
            'trainer_stats': self.trainer.get_training_statistics(),
            'config': self.config
        }
    
    # Persistence Methods - Delegate to persistence module
    
    def save_weights(self, filename: str = "dirac_nn_weights.npz") -> None:
        """
        Save neural network weights to a file in the project directory.
        
        Args:
            filename (str): Name of the file to save weights to. Defaults to "dirac_nn_weights.npz"
        """
        success = self.persistence.save_network_weights(filename, {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'network_architecture': [len(layer.neurons) for layer in self.network.layers],
            'config': self.config
        })
        if success:
            logger.info(f"Weights saved to {filename}")
        else:
            logger.error(f"Failed to save weights to {filename}")
    
    def load_weights(self, filename: str = "dirac_nn_weights.npz") -> bool:
        """
        Load neural network weights from a file in the project directory.
        
        Args:
            filename (str): Name of the file to load weights from. Defaults to "dirac_nn_weights.npz"
            
        Returns:
            bool: True if weights loaded successfully, False otherwise
        """
        try:
            metadata = self.persistence.load_network_weights(filename)
            logger.info(f"Successfully loaded weights from {filename}")
            if metadata:
                logger.info(f"Model trained on: {metadata.get('training_date', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False
    
    # Checkpoint Management Methods
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint information
        """
        return self.persistence.list_available_checkpoints()
    
    def load_checkpoint(self, checkpoint_filename: str = None, epoch: int = None) -> bool:
        """
        Load a specific checkpoint by filename or epoch number.
        
        Args:
            checkpoint_filename (str, optional): Specific checkpoint filename
            epoch (int, optional): Epoch number to load
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            result = self.persistence.load_checkpoint(checkpoint_filename, epoch)
            logger.info(f"Checkpoint loaded successfully")
            if 'metadata' in result:
                metadata = result['metadata']
                logger.info(f"Loaded from epoch {metadata.get('epoch', 'unknown')} with validation loss {metadata.get('validation_loss', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(self) -> bool:
        """
        Load the most recent checkpoint available.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            latest_checkpoint = self.persistence.find_latest_checkpoint()
            if latest_checkpoint is None:
                logger.warning("No checkpoints found")
                return False
            
            return self.load_checkpoint(latest_checkpoint['filename'])
        except Exception as e:
            logger.error(f"Failed to load latest checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_last_n (int): Number of recent checkpoints to keep
            
        Returns:
            int: Number of checkpoints removed
        """
        return self.persistence.cleanup_old_checkpoints(keep_last_n)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the refactored neural network with comprehensive training statistics.
    
    This example demonstrates:
    1. General-purpose training with comprehensive statistics and benchmarking
    2. Automatic model checkpointing and early stopping
    3. Performance analysis and acceleration factor measurement
    4. Detailed training monitoring and result saving
    """
    
    logger.info("NEURAL NETWORK FOR DIRAC POINT PREDICTION")
    logger.info("=" * 50)
    
    # Create the network (no parameters needed during initialization)
    nn = nn_dirac_point()
    
    # Check for existing trained model or checkpoints
    final_model_path = os.path.join(constants.PATH, "final_trained_model.npz")
    trained = False

    # First, try to load final trained model
    if os.path.exists(final_model_path):
        logger.info("Found existing trained model. Loading...")
        if nn.load_weights("final_trained_model.npz"):
            logger.info("Pre-trained model loaded successfully!")
            trained = True
        else:
            logger.warning("Failed to load pre-trained model.")

    # If no final model, try to resume from latest checkpoint
    resume_epoch = 0
    resume_patience_counter = 0
    if not trained:
        logger.info("Looking for checkpoints to resume training...")
        latest_checkpoint = nn.persistence.find_latest_checkpoint()
        if latest_checkpoint and nn.load_latest_checkpoint():
            resume_epoch = latest_checkpoint.get('epoch', 0)
            resume_patience_counter = latest_checkpoint.get('patience_counter', 0)
            logger.info(f"Resumed from checkpoint successfully!")
            logger.info(f"Loaded from epoch {resume_epoch} with validation loss {latest_checkpoint.get('validation_loss', 'unknown')}")
            logger.info(f"Resumed patience counter: {resume_patience_counter}")
            trained = False  # Continue training from checkpoint
        else:
            logger.info("No existing model or checkpoints found. Will train from scratch.")
            trained = False
    
    # Train using the comprehensive general training method
    if not trained:
        logger.info("Starting training...")
        
        try:
            # Run the general training method with production settings
            # Adjust epochs to account for already completed training
            remaining_epochs = constants.DEFAULT_MAX_EPOCHS - resume_epoch
            if remaining_epochs <= 0:
                logger.info(f"Training already completed {resume_epoch} epochs, target was {constants.DEFAULT_MAX_EPOCHS}")
                remaining_epochs = 50  # Train a bit more if desired

            training_results = nn.train_general(
                epochs=remaining_epochs,
                batch_size=constants.DEFAULT_TRAINING_BATCH_SIZE,
                validation_split=constants.DEFAULT_VALIDATION_SPLIT,
                save_checkpoints=True,
                checkpoint_interval=constants.DEFAULT_CHECKPOINT_INTERVAL,
                early_stopping_patience=constants.DEFAULT_EARLY_STOPPING_PATIENCE,
                start_epoch=resume_epoch,
                resume_patience_counter=resume_patience_counter
            )
            
            logger.info("Training completed successfully!")
            if 'total_training_time' in training_results:
                logger.info(f"   Time: {training_results['total_training_time']/60:.1f} min")
            if 'epochs_completed' in training_results:
                logger.info(f"   Epochs: {training_results['epochs_completed']}")
            if 'best_validation_loss' in training_results:
                logger.info(f"   Best loss: {training_results['best_validation_loss']:.6f}")
            
            trained = True
            
        except ValueError as e:
            logger.error(f"Training failed - no training data: {e}")
            trained = False
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            trained = False
    
    # Test predictions and analyze performance if we have a trained network
    if trained:
        logger.info("Testing predictions...")
        
        # Example test parameters for demonstration
        test_parameter_sets = [
            [5, 1, 1.0, 0.8, 0.5, 1.0],   # Small TBG system
            [7, 1, 1.1, 0.9, 0.6, 1.1],   # Medium TBG system  
            [13, 1, 1.2, 0.7, 0.4, 0.9],  # Larger TBG system
            [17, 1, 1.3, 0.8, 0.5, 1.0],  # Large TBG system
        ]
        
        for i, params in enumerate(test_parameter_sets, 1):
            try:
                k_x, k_y, velocity = nn.predict(params)
                logger.info(f"  Test {i}: k=({k_x:.4f}, {k_y:.4f}), v={velocity:.2f}")
            except Exception as e:
                logger.warning(f"  Test {i}: Failed - {e}")
        
        # Performance benchmarking
        try:
            benchmark_results = nn.benchmark_acceleration_factor(
                test_parameter_sets[:2], num_iterations=3
            )
            
            if "error" not in benchmark_results:
                logger.info(f"Network is {benchmark_results['acceleration_factor']:.1f}x faster than physics")
            
        except Exception:
            pass
        
        # Display statistics
        try:
            nn.log_training_statistics()
        except Exception:
            pass
            
    else:
        logger.warning("Training was not successful.")
        logger.info("Check Training_data/ folder and generate data if needed.")
    
