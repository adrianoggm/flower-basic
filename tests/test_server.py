"""Tests for the central server component."""

import json
from unittest.mock import MagicMock, Mock, patch

import flwr as fl
import numpy as np
import pytest
import torch


class TestMQTTFedAvg:
    """Test cases for the custom MQTT FedAvg strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        from model import ECGModel

        self.model = ECGModel()
        self.mock_mqtt_client = Mock()

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        from server import MQTTFedAvg

        strategy = MQTTFedAvg(
            model=self.model,
            mqtt_client=self.mock_mqtt_client,
            fraction_fit=1.0,
            min_fit_clients=1,
        )

        assert strategy.global_model is self.model
        assert strategy.mqtt is self.mock_mqtt_client
        assert hasattr(strategy, "param_names")
        assert len(strategy.param_names) > 0

    @patch("server.state_dict_to_numpy")
    def test_aggregate_fit_success(self, mock_state_dict_to_numpy):
        """Test successful aggregation and MQTT publishing."""
        from server import MQTTFedAvg

        # Setup strategy
        strategy = MQTTFedAvg(
            model=self.model,
            mqtt_client=self.mock_mqtt_client,
            fraction_fit=1.0,
            min_fit_clients=1,
        )

        # Mock the parent aggregate_fit method
        mock_params = Mock()
        mock_params.tensors = [
            fl.common.ndarray_to_bytes(np.random.randn(16, 1, 5).astype(np.float32)),
            fl.common.ndarray_to_bytes(np.random.randn(16).astype(np.float32)),
        ]

        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=mock_params
        ):
            mock_state_dict_to_numpy.return_value = {"param1": [1, 2], "param2": [3, 4]}

            # Create mock results
            mock_client_proxy = Mock()
            mock_fit_res = Mock()
            results = [(mock_client_proxy, mock_fit_res)]

            # Call aggregate_fit
            result = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            # Should return the aggregated parameters
            assert result is not None

            # Should publish to MQTT
            self.mock_mqtt_client.publish.assert_called_once()

            # Check the published message
            publish_call = self.mock_mqtt_client.publish.call_args
            topic, payload = publish_call[0]

            assert topic == "fl/global_model"

            # Parse the payload
            parsed_payload = json.loads(payload)
            assert "round" in parsed_payload
            assert "global_weights" in parsed_payload
            assert parsed_payload["round"] == 1

    def test_aggregate_fit_no_mqtt(self):
        """Test aggregation when MQTT client is None."""
        from server import MQTTFedAvg

        # Initialize strategy without MQTT client
        strategy = MQTTFedAvg(
            model=self.model, mqtt_client=None, fraction_fit=1.0, min_fit_clients=1
        )

        # Mock successful aggregation
        mock_params = Mock()
        mock_params.tensors = [
            fl.common.ndarray_to_bytes(np.random.randn(16, 1, 5).astype(np.float32))
        ]

        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=mock_params
        ):
            with patch("server.state_dict_to_numpy", return_value={}):
                mock_client_proxy = Mock()
                mock_fit_res = Mock()
                results = [(mock_client_proxy, mock_fit_res)]

                # Should not raise exception
                result = strategy.aggregate_fit(
                    server_round=1, results=results, failures=[]
                )

                assert result is not None

    def test_aggregate_fit_parent_returns_none(self):
        """Test behavior when parent aggregation returns None."""
        from server import MQTTFedAvg

        strategy = MQTTFedAvg(
            model=self.model,
            mqtt_client=self.mock_mqtt_client,
            fraction_fit=1.0,
            min_fit_clients=1,
        )

        # Mock parent returning None (aggregation failed)
        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=None
        ):
            result = strategy.aggregate_fit(server_round=1, results=[], failures=[])

            # Should return None
            assert result is None

            # Should not publish to MQTT
            self.mock_mqtt_client.publish.assert_not_called()

    def test_aggregate_fit_mqtt_exception(self):
        """Test handling of MQTT publishing exceptions."""
        from server import MQTTFedAvg

        strategy = MQTTFedAvg(
            model=self.model,
            mqtt_client=self.mock_mqtt_client,
            fraction_fit=1.0,
            min_fit_clients=1,
        )

        # Mock successful aggregation
        mock_params = Mock()
        mock_params.tensors = [
            fl.common.ndarray_to_bytes(np.random.randn(16, 1, 5).astype(np.float32))
        ]

        # Mock MQTT publish to raise exception
        self.mock_mqtt_client.publish.side_effect = Exception("MQTT Error")

        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=mock_params
        ):
            with patch("server.state_dict_to_numpy", return_value={}):
                mock_client_proxy = Mock()
                mock_fit_res = Mock()
                results = [(mock_client_proxy, mock_fit_res)]

                # Should handle exception gracefully
                try:
                    result = strategy.aggregate_fit(
                        server_round=1, results=results, failures=[]
                    )
                    # Should still return the aggregated parameters
                    assert result is not None
                except Exception as e:
                    pytest.fail(
                        f"Should handle MQTT exception gracefully, but raised: {e}"
                    )

    def test_parameter_conversion_flow(self):
        """Test the parameter conversion flow in aggregate_fit."""
        from server import MQTTFedAvg

        strategy = MQTTFedAvg(
            model=self.model,
            mqtt_client=self.mock_mqtt_client,
            fraction_fit=1.0,
            min_fit_clients=1,
        )

        # Create realistic parameter data
        sample_params = [
            np.random.randn(16, 1, 5).astype(np.float32),
            np.random.randn(16).astype(np.float32),
        ]

        # Mock Flower Parameters object
        mock_params = Mock()
        mock_params.tensors = [
            fl.common.ndarray_to_bytes(param) for param in sample_params
        ]

        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=mock_params
        ):
            mock_client_proxy = Mock()
            mock_fit_res = Mock()
            results = [(mock_client_proxy, mock_fit_res)]

            result = strategy.aggregate_fit(
                server_round=1, results=results, failures=[]
            )

            # Should succeed
            assert result is not None

            # Should publish to MQTT
            self.mock_mqtt_client.publish.assert_called_once()


class TestServerMain:
    """Test cases for the main server function."""

    @patch("server.fl.server.start_server")
    @patch("server.mqtt.Client")
    def test_main_function_mqtt_success(self, mock_mqtt_class, mock_start_server):
        """Test main function with successful MQTT connection."""
        from server import main

        # Mock MQTT client
        mock_mqtt_client = Mock()
        mock_mqtt_class.return_value = mock_mqtt_client

        # Call main function
        main()

        # Check MQTT setup
        mock_mqtt_class.assert_called_once()
        mock_mqtt_client.connect.assert_called_once()
        mock_mqtt_client.loop_start.assert_called_once()

        # Check Flower server startup
        mock_start_server.assert_called_once()

        # Check server config
        call_args = mock_start_server.call_args
        assert call_args[1]["server_address"] == "0.0.0.0:8080"
        assert hasattr(call_args[1]["config"], "num_rounds")
        assert call_args[1]["config"].num_rounds == 3

    @patch("server.fl.server.start_server")
    @patch("server.mqtt.Client")
    def test_main_function_mqtt_failure(self, mock_mqtt_class, mock_start_server):
        """Test main function with MQTT connection failure."""
        from server import main

        # Mock MQTT client with connection failure
        mock_mqtt_client = Mock()
        mock_mqtt_client.connect.side_effect = Exception("Connection failed")
        mock_mqtt_class.return_value = mock_mqtt_client

        # Should handle MQTT failure gracefully
        try:
            main()
        except Exception as e:
            pytest.fail(f"Should handle MQTT failure gracefully, but raised: {e}")

        # Should still start Flower server
        mock_start_server.assert_called_once()

        # Strategy should be initialized with None MQTT client
        call_args = mock_start_server.call_args
        strategy = call_args[1]["strategy"]
        assert strategy.mqtt is None

    @patch("server.fl.server.start_server")
    def test_strategy_configuration(self, mock_start_server):
        """Test that the strategy is configured correctly."""
        from server import main

        with patch("server.mqtt.Client") as mock_mqtt_class:
            mock_mqtt_client = Mock()
            mock_mqtt_class.return_value = mock_mqtt_client

            main()

            # Check strategy configuration
            call_args = mock_start_server.call_args
            strategy = call_args[1]["strategy"]

            # Check strategy type
            from server import MQTTFedAvg

            assert isinstance(strategy, MQTTFedAvg)

            # Check strategy parameters
            assert hasattr(strategy, "global_model")
            assert hasattr(strategy, "mqtt")
            assert hasattr(strategy, "param_names")


class TestServerIntegration:
    """Integration tests for server components."""

    def test_mqtt_message_format(self):
        """Test that MQTT messages follow expected format."""
        from model import ECGModel
        from server import MQTTFedAvg

        model = ECGModel()
        mock_mqtt = Mock()

        strategy = MQTTFedAvg(
            model=model, mqtt_client=mock_mqtt, fraction_fit=1.0, min_fit_clients=1
        )

        # Create sample aggregated parameters
        sample_params = [
            np.random.randn(16, 1, 5).astype(np.float32),
            np.random.randn(16).astype(np.float32),
        ]

        mock_params = Mock()
        mock_params.tensors = [
            fl.common.ndarray_to_bytes(param) for param in sample_params
        ]

        with patch.object(
            fl.server.strategy.FedAvg, "aggregate_fit", return_value=mock_params
        ):
            mock_client = Mock()
            mock_result = Mock()
            results = [(mock_client, mock_result)]

            strategy.aggregate_fit(1, results, [])

            # Extract published message
            publish_call = mock_mqtt.publish.call_args
            topic, payload = publish_call[0]

            # Parse and validate message format
            message = json.loads(payload)

            assert "round" in message
            assert "global_weights" in message
            assert isinstance(message["round"], int)
            assert isinstance(message["global_weights"], dict)

            # Check that weights can be converted back
            for param_name, param_values in message["global_weights"].items():
                assert isinstance(param_values, list)
                # Should be convertible back to numpy
                np.array(param_values)

    def test_error_handling_edge_cases(self):
        """Test handling of various edge cases and errors."""
        from model import ECGModel
        from server import MQTTFedAvg

        model = ECGModel()
        mock_mqtt = Mock()

        strategy = MQTTFedAvg(
            model=model, mqtt_client=mock_mqtt, fraction_fit=1.0, min_fit_clients=1
        )

        # Test with empty results
        result = strategy.aggregate_fit(1, [], [])
        # Should handle gracefully (depends on parent implementation)

        # Test with malformed results
        with patch.object(
            fl.server.strategy.FedAvg,
            "aggregate_fit",
            side_effect=Exception("Aggregation error"),
        ):
            try:
                strategy.aggregate_fit(1, [], [])
            except Exception:
                # Expected - let parent handle the error
                pass
