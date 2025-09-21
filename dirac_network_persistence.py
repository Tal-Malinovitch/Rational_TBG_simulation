"""
File management and persistence utilities for Dirac point neural networks.

This module handles saving/loading neural network weights, training data management,
and result storage. Integrates with existing project utilities like simulation_data_loader
and stats for consistent file handling across the project.

Classes:
    dirac_network_persistence: Handles all file I/O operations for neural networks
"""

import constants
from constants import np, os, csv, logging, glob
from constants import List, Dict, Optional, Union, Any, Tuple
from neural_network_base import neural_network
from simulation_data_loader import simulation_data_analyzer
from stats import statistics

# Configure logging
logger = logging.getLogger(__name__)

# Default persistence configuration
default_persistence_config = {
    'auto_backup': True,
    'backup_versions': 3,
    'compression_level': 6,
    'verify_integrity': True
}


class dirac_network_persistence:
    """
    Handles file persistence operations for Dirac point neural networks.
    
    This class provides comprehensive file management including network weight
    saving/loading, training data management, and result storage. Integrates with
    existing project utilities for consistent file handling.
    
    Attributes:
        persistence_config (dict): File management configuration
        base_path (str): Base directory for all file operations
        current_network (Optional[neural_network]): Network for file operations
    """
    
    def __init__(self, base_path: Optional[str] = None, persistence_config: Optional[dict] = None) -> None:
        """
        Initialize the Dirac network persistence manager.
        
        Args:
            base_path (str, optional): Base directory for file operations.
                Uses constants.PATH if not provided.
            persistence_config (dict, optional): Persistence configuration.
                Uses default_persistence_config if not provided.
                
        Raises:
            constants.physics_parameter_error: If initialization fails
        """
        try:
            self.base_path = base_path or constants.PATH
            self.persistence_config = persistence_config or default_persistence_config.copy()
            self._validate_persistence_config()
            
            # Ensure base path exists
            os.makedirs(self.base_path, exist_ok=True)
            
            # Network reference for operations
            self.current_network: Optional[neural_network] = None
            
            logger.info(f"dirac_network_persistence initialized with base_path: {self.base_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dirac_network_persistence: {str(e)}")
            raise constants.physics_parameter_error(f"Persistence initialization failed: {str(e)}")
    
    def _validate_persistence_config(self) -> None:
        """
        Validate persistence configuration parameters.
        
        Raises:
            constants.physics_parameter_error: If configuration is invalid
        """
        if not isinstance(self.persistence_config, dict):
            raise constants.physics_parameter_error("persistence_config must be a dictionary")
        
        # Validate backup versions if specified
        if 'backup_versions' in self.persistence_config:
            backup_versions = self.persistence_config['backup_versions']
            if not isinstance(backup_versions, int) or backup_versions < 0:
                raise constants.physics_parameter_error("backup_versions must be non-negative integer")
    
    def set_network(self, network: neural_network) -> None:
        """
        Set the network for file operations.
        
        Args:
            network (neural_network): Neural network for persistence operations
        """
        self.current_network = network
        logger.info("Set network for persistence operations")
    
    def save_network_weights(self, filename: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save neural network weights to file with optional metadata.
        
        Args:
            filename (str): Filename for saving weights (without path)
            metadata (dict, optional): Additional metadata to save with weights
            
        Returns:
            bool: True if saved successfully, False otherwise
            
        Raises:
            constants.physics_parameter_error: If no network set or save fails
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network set. Call set_network() first.")
        
        try:
            # Ensure filename is in base path
            if not os.path.isabs(filename):
                filepath = os.path.join(self.base_path, filename)
            else:
                filepath = filename
            
            # Create backup if enabled
            if self.persistence_config.get('auto_backup', True) and os.path.exists(filepath):
                self._create_backup(filepath)
            
            weights_data = {}
            
            # Save weights from all layers (skip input layer which is dummy)
            for layer_idx, layer in enumerate(self.current_network.layers[1:], 1):
                layer_weights = []
                for neuron_idx, neuron in enumerate(layer.neurons):
                    neuron_weights = []
                    for input_neuron, weight in neuron.inputs:
                        neuron_weights.append(weight)
                    layer_weights.append(neuron_weights)
                
                weights_data[f'layer_{layer_idx}_weights'] = np.array(layer_weights)
            
            # Save network architecture metadata
            weights_data['num_layers'] = len(self.current_network.layers)
            weights_data['layer_sizes'] = [len(layer.neurons) for layer in self.current_network.layers]
            
            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in weights_data:  # Don't overwrite architecture data
                        weights_data[f'metadata_{key}'] = value
            
            # Save with compression
            compression_level = self.persistence_config.get('compression_level', 6)
            if compression_level > 0:
                np.savez_compressed(filepath, **weights_data)
            else:
                np.savez(filepath, **weights_data)
            
            # Verify integrity if enabled
            if self.persistence_config.get('verify_integrity', True):
                if not self._verify_file_integrity(filepath):
                    logger.error(f"File integrity verification failed: {filepath}")
                    return False
            
            logger.info(f"Network weights saved: {os.path.basename(filepath)}")
            logger.debug(f"Saved {len(weights_data)-2} weight matrices")  # -2 for metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save network weights: {str(e)}")
            return False
    
    def load_network_weights(self, filename: str, verify_compatibility: bool = True) -> Dict[str, Any]:
        """
        Load neural network weights from file with compatibility checking.
        
        Args:
            filename (str): Filename to load weights from
            verify_compatibility (bool): Whether to verify network compatibility
            
        Returns:
            Dict[str, Any]: Loading results and metadata
            
        Raises:
            constants.physics_parameter_error: If loading fails or incompatible
        """
        if self.current_network is None:
            raise constants.physics_parameter_error("No network set. Call set_network() first.")
        
        try:
            # Ensure filename is in base path
            if not os.path.isabs(filename):
                filepath = os.path.join(self.base_path, filename)
            else:
                filepath = filename
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Weight file not found: {filepath}")
            
            # Load weights data
            weights_data = np.load(filepath)
            
            # Extract metadata
            metadata = {}
            for key in weights_data.files:
                if key.startswith('metadata_'):
                    metadata[key[9:]] = weights_data[key]  # Remove 'metadata_' prefix
            
            # Verify network compatibility
            if verify_compatibility:
                if not self._verify_network_compatibility(weights_data):
                    raise constants.physics_parameter_error("Network architecture incompatible with saved weights")
            
            # Load weights into network
            loaded_layers = 0
            for layer_idx, layer in enumerate(self.current_network.layers[1:], 1):
                weight_key = f'layer_{layer_idx}_weights'
                
                if weight_key not in weights_data:
                    logger.warning(f"Missing weights for layer {layer_idx}")
                    continue
                
                layer_weights = weights_data[weight_key]
                
                # Set weights for each neuron in this layer
                for neuron_idx, neuron in enumerate(layer.neurons):
                    if neuron_idx >= len(layer_weights):
                        logger.warning(f"Insufficient weights for layer {layer_idx}, neuron {neuron_idx}")
                        continue
                    
                    neuron_weights = layer_weights[neuron_idx]
                    
                    # Update connection weights
                    for input_idx, (input_neuron, old_weight) in enumerate(neuron.inputs):
                        if input_idx >= len(neuron_weights):
                            continue
                        
                        new_weight = float(neuron_weights[input_idx])
                        neuron.change_weight(input_neuron, new_weight)
                        
                        # Update ADAM optimizer weight
                        if hasattr(neuron, 'Adam_corrector') and input_idx < len(neuron.Adam_corrector):
                            neuron.Adam_corrector[input_idx].weight = new_weight
                
                loaded_layers += 1
            
            weights_data.close()  # Clean up numpy file handle
            
            result = {
                'success': True,
                'loaded_layers': loaded_layers,
                'total_layers': len(self.current_network.layers) - 1,  # Exclude input layer
                'metadata': metadata,
                'filepath': filepath
            }
            
            logger.info(f"Network weights loaded: {os.path.basename(filepath)}")
            logger.debug(f"Loaded {loaded_layers} layers with metadata: {list(metadata.keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load network weights: {str(e)}")
            raise constants.physics_parameter_error(f"Weight loading failed: {str(e)}")
    
    def save_training_results(self, results: Dict[str, Any], filename: str) -> bool:
        """
        Save comprehensive training results to file.
        
        Args:
            results (Dict[str, Any]): Training results dictionary
            filename (str): Output filename
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            filepath = os.path.join(self.base_path, filename) if not os.path.isabs(filename) else filename
            
            # Create backup if exists
            if self.persistence_config.get('auto_backup', True) and os.path.exists(filepath):
                self._create_backup(filepath)
            
            # Save as compressed numpy file
            np.savez_compressed(filepath, **results)
            
            logger.info(f"Training results saved: {os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training results: {str(e)}")
            return False
    
    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoint files with their metadata.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint information dictionaries
        """
        checkpoints = []
        
        try:
            # Look for checkpoint files in base path
            checkpoint_pattern = os.path.join(self.base_path, "checkpoint_epoch_*.npz")
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            # Also look for best model
            best_model_path = os.path.join(self.base_path, "best_model.npz")
            if os.path.exists(best_model_path):
                checkpoint_files.append(best_model_path)
            
            for filepath in sorted(checkpoint_files):
                try:
                    # Load metadata without loading full weights
                    data = np.load(filepath, allow_pickle=True)
                    
                    checkpoint_info = {
                        'filepath': filepath,
                        'filename': os.path.basename(filepath),
                        'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
                        'modified_time': os.path.getmtime(filepath)
                    }
                    
                    # Extract metadata
                    for key in data.files:
                        if key.startswith('metadata_'):
                            metadata_key = key[9:]  # Remove 'metadata_' prefix
                            checkpoint_info[metadata_key] = data[key].item()
                    
                    checkpoints.append(checkpoint_info)
                    data.close()
                    
                except Exception as e:
                    logger.warning(f"Could not read checkpoint metadata from {filepath}: {e}")
            
            logger.info(f"Found {len(checkpoints)} available checkpoints")
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []
    
    def load_checkpoint(self, checkpoint_filename: str = None, epoch: int = None) -> Dict[str, Any]:
        """
        Load a specific checkpoint by filename or epoch number.
        
        Args:
            checkpoint_filename (str, optional): Specific checkpoint filename
            epoch (int, optional): Epoch number to load (finds checkpoint_epoch_X.npz)
            
        Returns:
            Dict[str, Any]: Checkpoint metadata and loading results
            
        Raises:
            constants.physics_parameter_error: If checkpoint loading fails
        """
        if checkpoint_filename is None and epoch is None:
            raise constants.physics_parameter_error("Must specify either checkpoint_filename or epoch")
        
        try:
            # Determine filename
            if checkpoint_filename is None:
                checkpoint_filename = f"checkpoint_epoch_{epoch}.npz"
            
            # Load the checkpoint
            result = self.load_network_weights(checkpoint_filename)
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_filename}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise constants.physics_parameter_error(f"Checkpoint loading failed: {str(e)}")
    
    def find_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Find and return information about the latest checkpoint.
        
        Returns:
            Optional[Dict[str, Any]]: Latest checkpoint info or None if no checkpoints found
        """
        checkpoints = self.list_available_checkpoints()
        
        if not checkpoints:
            return None
        
        # Include both training checkpoints AND best_model for resuming
        resumable_checkpoints = [cp for cp in checkpoints if
                               ('checkpoint_epoch_' in cp['filename'] or cp['filename'] == 'best_model.npz')]

        if not resumable_checkpoints:
            return None
        
        # Sort by epoch number, treating best_model as a valid resume point
        def extract_epoch(checkpoint) -> int:
            try:
                # For best_model.npz, use the epoch from metadata if available
                epoch = checkpoint.get('epoch', 0)
                if epoch == 0 and checkpoint['filename'] == 'best_model.npz':
                    # If best_model has no epoch info, use modification time as fallback
                    logger.info(f"best_model.npz found without epoch info, using as resume point")
                    return checkpoint.get('modified_time', 0)
                return epoch
            except:
                return 0

        latest_checkpoint = max(resumable_checkpoints, key=extract_epoch)
        logger.info(f"Latest checkpoint: {latest_checkpoint['filename']} at epoch {latest_checkpoint.get('epoch', 'unknown')}")
        
        return latest_checkpoint
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_last_n (int): Number of recent checkpoints to keep
            
        Returns:
            int: Number of checkpoints removed
        """
        try:
            checkpoints = self.list_available_checkpoints()
            
            # Filter for training checkpoints only (preserve best_model.npz)
            training_checkpoints = [cp for cp in checkpoints if 'checkpoint_epoch_' in cp['filename']]
            
            if len(training_checkpoints) <= keep_last_n:
                logger.info(f"No checkpoint cleanup needed. Found {len(training_checkpoints)} checkpoints, keeping {keep_last_n}")
                return 0
            
            # Sort by epoch and remove oldest ones
            def extract_epoch(checkpoint) -> int:
                return checkpoint.get('epoch', 0)
            
            sorted_checkpoints = sorted(training_checkpoints, key=extract_epoch, reverse=True)
            checkpoints_to_remove = sorted_checkpoints[keep_last_n:]
            
            removed_count = 0
            for checkpoint in checkpoints_to_remove:
                try:
                    os.remove(checkpoint['filepath'])
                    removed_count += 1
                    logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['filename']}: {e}")
            
            logger.info(f"Checkpoint cleanup completed: removed {removed_count} old checkpoints")
            return removed_count
            
        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {str(e)}")
            return 0
    
    def load_training_data_using_analyzer(self, data_folder: Optional[str] = None,
                                        remove_duplicates: bool = True) -> simulation_data_analyzer:
        """
        Load training data using existing project utility.
        
        Args:
            data_folder (str, optional): Path to training data folder
            remove_duplicates (bool): Whether to remove duplicate data points
            
        Returns:
            simulation_data_analyzer: Loaded data analyzer from project utility
            
        Raises:
            ValueError: If no training data found
        """
        if data_folder is None:
            data_folder = os.path.join(self.base_path, "Training_data")
        
        try:
            # Use existing project utility
            analyzer = simulation_data_analyzer(data_folder)
            
            # Find and load CSV files
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {data_folder}")
            
            analyzer.load_multiple_csv_files(csv_files, remove_duplicates=remove_duplicates)
            
            if not analyzer.data_points:
                raise ValueError("No valid training data loaded")
            
            logger.info(f"Loaded training data using analyzer: {len(analyzer.data_points)} data points")
            
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def save_statistics_using_utility(self, stats_obj: statistics, filename: str) -> bool:
        """
        Save statistics using existing project utility.
        
        Args:
            stats_obj (statistics): Statistics object to save
            filename (str): Output filename
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            filepath = os.path.join(self.base_path, filename) if not os.path.isabs(filename) else filename
            
            # Use existing project utility
            stats_obj.save_statistics(filepath)
            
            logger.info(f"Statistics saved using utility: {os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {str(e)}")
            return False
    
    def export_predictions_to_csv(self, predictions: List[Dict[str, Any]], filename: str) -> bool:
        """
        Export prediction results to CSV format.
        
        Args:
            predictions (List[Dict[str, Any]]): List of prediction results
            filename (str): Output CSV filename
            
        Returns:
            bool: True if exported successfully, False otherwise
        """
        if not predictions:
            logger.warning("No predictions to export")
            return False
        
        try:
            filepath = os.path.join(self.base_path, filename) if not os.path.isabs(filename) else filename
            
            # Create backup if exists
            if self.persistence_config.get('auto_backup', True) and os.path.exists(filepath):
                self._create_backup(filepath)
            
            # Extract fieldnames from first prediction
            fieldnames = list(predictions[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(predictions)
            
            logger.info(f"Predictions exported to CSV: {os.path.basename(filepath)}")
            logger.debug(f"Exported {len(predictions)} predictions with fields: {fieldnames}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export predictions to CSV: {str(e)}")
            return False
    
    def _create_backup(self, filepath: str) -> None:
        """Create backup of existing file."""
        try:
            backup_versions = self.persistence_config.get('backup_versions', 3)
            if backup_versions <= 0:
                return
            
            base_name = os.path.splitext(filepath)[0]
            extension = os.path.splitext(filepath)[1]
            
            # Rotate existing backups
            for i in range(backup_versions - 1, 0, -1):
                old_backup = f"{base_name}.backup_{i}{extension}"
                new_backup = f"{base_name}.backup_{i+1}{extension}"
                
                if os.path.exists(old_backup):
                    if os.path.exists(new_backup):
                        os.remove(new_backup)
                    os.rename(old_backup, new_backup)
            
            # Create new backup
            backup_path = f"{base_name}.backup_1{extension}"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.debug(f"Created backup: {os.path.basename(backup_path)}")
            
        except Exception as e:
            logger.warning(f"Failed to create backup for {filepath}: {str(e)}")
    
    def _verify_file_integrity(self, filepath: str) -> bool:
        """Verify file integrity by attempting to reload."""
        try:
            test_data = np.load(filepath)
            test_data.close()
            return True
        except Exception:
            return False
    
    def _verify_network_compatibility(self, weights_data) -> bool:
        """Verify loaded weights are compatible with current network."""
        try:
            if 'num_layers' not in weights_data or 'layer_sizes' not in weights_data:
                return False
            
            saved_num_layers = int(weights_data['num_layers'])
            saved_layer_sizes = list(weights_data['layer_sizes'])
            
            current_num_layers = len(self.current_network.layers)
            current_layer_sizes = [len(layer.neurons) for layer in self.current_network.layers]
            
            return (saved_num_layers == current_num_layers and 
                   saved_layer_sizes == current_layer_sizes)
            
        except Exception:
            return False
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved model files in the base directory.
        
        Returns:
            List[Dict[str, Any]]: List of model file information
        """
        model_files = []
        
        try:
            for filename in os.listdir(self.base_path):
                if filename.endswith('.npz') and not filename.startswith('.'):
                    filepath = os.path.join(self.base_path, filename)
                    
                    try:
                        # Get file statistics
                        stat_info = os.stat(filepath)
                        
                        # Try to load metadata
                        metadata = {}
                        try:
                            weights_data = np.load(filepath)
                            for key in weights_data.files:
                                if key.startswith('metadata_'):
                                    metadata[key[9:]] = str(weights_data[key])
                            weights_data.close()
                        except Exception:
                            pass
                        
                        model_files.append({
                            'filename': filename,
                            'size_mb': stat_info.st_size / (1024 * 1024),
                            'modified_time': stat_info.st_mtime,
                            'metadata': metadata
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not read model file {filename}: {str(e)}")
                        continue
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x['modified_time'], reverse=True)
            
            return model_files
            
        except Exception as e:
            logger.error(f"Failed to list saved models: {str(e)}")
            return []
    
    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """
        Clean up old backup files.
        
        Args:
            max_age_days (int): Maximum age of backup files to keep
            
        Returns:
            int: Number of backup files removed
        """
        import time as time_module
        
        removed_count = 0
        cutoff_time = time_module.time() - (max_age_days * 24 * 60 * 60)
        
        try:
            for filename in os.listdir(self.base_path):
                if '.backup_' in filename:
                    filepath = os.path.join(self.base_path, filename)
                    
                    try:
                        if os.path.getctime(filepath) < cutoff_time:
                            os.remove(filepath)
                            removed_count += 1
                            logger.debug(f"Removed old backup: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove backup {filename}: {str(e)}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old backup files")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {str(e)}")
            return 0


# Convenience functions following project patterns
def save_network_with_metadata(network: neural_network, filename: str, 
                              metadata: Dict[str, Any], base_path: Optional[str] = None) -> bool:
    """
    Convenience function to save network with metadata.
    
    Args:
        network (neural_network): Network to save
        filename (str): Output filename
        metadata (Dict[str, Any]): Metadata to include
        base_path (str, optional): Base directory path
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    persistence = dirac_network_persistence(base_path)
    persistence.set_network(network)
    return persistence.save_network_weights(filename, metadata)


def load_training_data_quick(data_folder: Optional[str] = None) -> simulation_data_analyzer:
    """
    Quick training data loading using project utilities.
    
    Args:
        data_folder (str, optional): Training data folder path
        
    Returns:
        simulation_data_analyzer: Loaded data analyzer
        
    Raises:
        ValueError: If no training data found
    """
    persistence = dirac_network_persistence()
    return persistence.load_training_data_using_analyzer(data_folder)