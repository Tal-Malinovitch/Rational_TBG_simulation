#!/usr/bin/env python3
"""
Test script to verify training stability improvements.

This script tests the new gradient clipping, adaptive learning rate,
and frequent checkpointing features added to prevent gradient explosion.
"""

import constants
from constants import np, logging
from NN_Dirac_point import nn_dirac_point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_stability_config():
    """Test that all training stability constants are properly configured."""
    logger.info("Testing training stability configuration...")

    # Test that gradient clipping constant is set
    assert hasattr(constants, 'GRADIENT_CLIP_VALUE'), "GRADIENT_CLIP_VALUE not found in constants"
    assert constants.GRADIENT_CLIP_VALUE == 5.0, f"Expected GRADIENT_CLIP_VALUE=5.0, got {constants.GRADIENT_CLIP_VALUE}"

    # Test that checkpoint interval is updated to 20
    assert constants.DEFAULT_CHECKPOINT_INTERVAL == 20, f"Expected checkpoint interval=20, got {constants.DEFAULT_CHECKPOINT_INTERVAL}"

    # Test that loss explosion detection constants are set
    assert hasattr(constants, 'LOSS_EXPLOSION_THRESHOLD'), "LOSS_EXPLOSION_THRESHOLD not found"
    assert hasattr(constants, 'LEARNING_RATE_REDUCTION_FACTOR'), "LEARNING_RATE_REDUCTION_FACTOR not found"
    assert hasattr(constants, 'MIN_LEARNING_RATE'), "MIN_LEARNING_RATE not found"
    assert hasattr(constants, 'LOSS_HISTORY_WINDOW'), "LOSS_HISTORY_WINDOW not found"

    logger.info("✓ All training stability constants properly configured")

    # Test neural network initialization with new config
    logger.info("Testing neural network initialization...")
    nn = nn_dirac_point()

    # Check that trainer has the new stability features
    trainer = nn.trainer
    assert hasattr(trainer, 'current_learning_rate'), "Trainer missing current_learning_rate"
    assert hasattr(trainer, 'loss_history'), "Trainer missing loss_history"
    assert hasattr(trainer, 'learning_rate_reductions'), "Trainer missing learning_rate_reductions"

    # Check that trainer config includes new parameters
    config = trainer.training_config
    assert 'gradient_clip_value' in config, "gradient_clip_value missing from trainer config"
    assert 'loss_explosion_threshold' in config, "loss_explosion_threshold missing from trainer config"
    assert 'learning_rate_reduction_factor' in config, "learning_rate_reduction_factor missing from trainer config"

    logger.info("✓ Neural network and trainer properly initialized with stability features")

    # Test that gradient clipping value is correct
    assert config['gradient_clip_value'] == 5.0, f"Expected gradient_clip_value=5.0, got {config['gradient_clip_value']}"

    # Test checkpoint interval
    assert config['checkpoint_interval'] == 20, f"Expected checkpoint_interval=20, got {config['checkpoint_interval']}"

    logger.info("✓ All configuration values correct")

def test_loss_explosion_detection():
    """Test loss explosion detection functionality."""
    logger.info("Testing loss explosion detection...")

    nn = nn_dirac_point()
    trainer = nn.trainer

    # Test normal loss - should not trigger explosion detection
    assert not trainer._detect_loss_explosion(1.0), "False positive on normal loss"
    assert not trainer._detect_loss_explosion(2.0), "False positive on normal loss"
    assert not trainer._detect_loss_explosion(1.5), "False positive on normal loss"

    # Test explosive loss - should trigger detection
    assert trainer._detect_loss_explosion(200.0), "Failed to detect explosive loss"

    logger.info("✓ Loss explosion detection working correctly")

def test_learning_rate_adaptation():
    """Test adaptive learning rate functionality."""
    logger.info("Testing adaptive learning rate...")

    nn = nn_dirac_point()
    trainer = nn.trainer

    initial_lr = trainer.current_learning_rate
    logger.info(f"Initial learning rate: {initial_lr:.6f}")

    # Test learning rate reduction
    success = trainer._reduce_learning_rate()
    assert success, "Learning rate reduction failed"

    new_lr = trainer.current_learning_rate
    expected_lr = initial_lr * constants.LEARNING_RATE_REDUCTION_FACTOR
    assert abs(new_lr - expected_lr) < 1e-10, f"Expected LR {expected_lr:.6f}, got {new_lr:.6f}"

    logger.info(f"Learning rate after reduction: {new_lr:.6f}")
    logger.info("✓ Adaptive learning rate working correctly")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING TRAINING STABILITY IMPROVEMENTS")
    logger.info("=" * 60)

    try:
        test_training_stability_config()
        test_loss_explosion_detection()
        test_learning_rate_adaptation()

        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED! Training stability improvements verified.")
        logger.info("=" * 60)
        logger.info("Summary of improvements:")
        logger.info(f"✓ Gradient clipping: {constants.GRADIENT_CLIP_VALUE}")
        logger.info(f"✓ Checkpoint interval: {constants.DEFAULT_CHECKPOINT_INTERVAL} epochs")
        logger.info(f"✓ Loss explosion threshold: {constants.LOSS_EXPLOSION_THRESHOLD}")
        logger.info(f"✓ Learning rate reduction factor: {constants.LEARNING_RATE_REDUCTION_FACTOR}")
        logger.info(f"✓ Minimum learning rate: {constants.MIN_LEARNING_RATE}")

    except Exception as e:
        logger.error(f"❌ TEST FAILED: {str(e)}")
        raise