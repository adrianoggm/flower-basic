"""Tests for the ECG CNN model."""

import numpy as np
import pytest
import torch

from flower_basic.model import ECGModel, get_parameters, set_parameters


class TestECGModel:
    """Test cases for the ECGModel CNN."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ECGModel()
        self.batch_size = 4
        self.seq_len = 140
        self.input_tensor = torch.randn(self.batch_size, 1, self.seq_len)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        assert isinstance(self.model, ECGModel)
        assert hasattr(self.model, "conv1")
        assert hasattr(self.model, "conv2")
        assert hasattr(self.model, "fc1")
        assert hasattr(self.model, "fc2")

    def test_forward_pass(self):
        """Test forward pass with correct input dimensions."""
        output = self.model(self.input_tensor)

        # Check output shape
        assert output.shape == (self.batch_size, 1)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        single_input = torch.randn(1, 1, self.seq_len)
        output = self.model(single_input)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_invalid_input_dimensions(self):
        """Test that invalid input dimensions raise appropriate errors."""
        # Wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError)):
            self.model(torch.randn(self.batch_size, self.seq_len))

        # Wrong sequence length (too short for conv operations)
        with pytest.raises((RuntimeError, ValueError)):
            self.model(torch.randn(self.batch_size, 1, 10))

    def test_get_parameters(self):
        """Test parameter extraction."""
        params = get_parameters(self.model)

        assert isinstance(params, list)
        assert len(params) > 0

        # Check that all parameters are numpy arrays
        for param in params:
            assert isinstance(param, np.ndarray)
            assert param.dtype == np.float32

    def test_set_parameters(self):
        """Test parameter setting."""
        # Get original first parameter value
        original_conv1_weight = self.model.conv1.weight.data.clone()

        # Get all parameters
        original_params = get_parameters(self.model)

        # Create completely different parameters (ones instead of random values)
        modified_params = [np.ones_like(param) for param in original_params]

        # Set modified parameters
        set_parameters(self.model, modified_params)

        # Check directly from model that parameters changed
        new_conv1_weight = self.model.conv1.weight.data

        # Verify the change is real
        assert not torch.equal(
            original_conv1_weight, new_conv1_weight
        ), "Model parameters should have changed after setting new values"

        # Additional verification: new weights should be all ones
        assert torch.allclose(
            new_conv1_weight, torch.ones_like(new_conv1_weight)
        ), "New parameters should be all ones"

    def test_parameter_shapes_consistency(self):
        """Test that parameter shapes are consistent."""
        params = get_parameters(self.model)

        # Set the same parameters back
        set_parameters(self.model, params)

        # Get parameters again
        params_after = get_parameters(self.model)

        # Check shapes are identical
        for orig, after in zip(params, params_after):
            assert orig.shape == after.shape
            np.testing.assert_array_almost_equal(orig, after, decimal=6)

    def test_model_training_mode(self):
        """Test model can switch between training and evaluation modes."""
        # Test training mode
        self.model.train()
        assert self.model.training

        # Test evaluation mode
        self.model.eval()
        assert not self.model.training

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        self.model.train()

        # Forward pass
        output = self.model(self.input_tensor)

        # Create dummy loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestParameterUtilities:
    """Test cases for parameter utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ECGModel()

    def test_parameter_conversion_roundtrip(self):
        """Test that parameter conversion is lossless."""
        # Get original state
        original_state = self.model.state_dict()

        # Convert to numpy and back
        numpy_params = get_parameters(self.model)
        set_parameters(self.model, numpy_params)
        final_state = self.model.state_dict()

        # Check that state is preserved
        for key in original_state:
            torch.testing.assert_close(original_state[key], final_state[key])

    def test_parameter_types(self):
        """Test parameter type consistency."""
        params = get_parameters(self.model)

        for param in params:
            assert isinstance(param, np.ndarray)
            assert param.dtype == np.float32

    def test_empty_parameters_handling(self):
        """Test handling of edge cases."""
        # Test with empty list
        empty_params = []

        # This should handle gracefully or raise appropriate error
        try:
            set_parameters(self.model, empty_params)
        except (ValueError, RuntimeError) as e:
            # Expected behavior for invalid input
            assert "parameters" in str(e).lower() or "mismatch" in str(e).lower()
