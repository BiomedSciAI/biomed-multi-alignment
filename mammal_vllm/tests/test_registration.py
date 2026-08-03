"""
tests/test_registration.py
--------------------------
Verify that the plugin entry-point and model classes are importable.
No GPU, no network, no model weights required.
"""

import pytest

class TestPluginRegistration:
    """Verify that the plugin entry-point and model class are importable."""

    def test_register_function_exists(self) -> None:
        """The registration callable must be importable from the package."""
        from vllm_mammal_plugin import register_mammal_model

        assert callable(register_mammal_model)

    def test_model_class_importable(self) -> None:
        """T5ForConditionalGeneration must expose load_weights and forward."""
        from vllm_mammal_plugin.mammal import T5ForConditionalGeneration

        assert T5ForConditionalGeneration is not None
        assert hasattr(T5ForConditionalGeneration, "load_weights")
        assert hasattr(T5ForConditionalGeneration, "forward")

    def test_config_class_importable(self) -> None:
        """MammalConfig must be importable and carry model_type='t5'."""
        from vllm_mammal_plugin.mammal import MammalConfig

        assert MammalConfig is not None
        assert MammalConfig.model_type == "t5"


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
