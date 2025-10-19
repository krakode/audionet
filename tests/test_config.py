"""Tests for AudioNet configuration."""

import pytest
from audionet.config import AudioNetConfig
from audionet.exceptions import ConfigurationError


class TestAudioNetConfig:
    """Test AudioNetConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AudioNetConfig()
        
        assert config.target_sr == 16000
        assert config.vad_frame_ms == 30
        assert config.vad_aggressiveness == 2
        assert config.min_voiced_seconds == 1.0
        assert config.min_voiced_ratio == 0.15
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AudioNetConfig(
            target_sr=8000,
            vad_aggressiveness=1,
            min_voiced_seconds=0.5
        )
        
        assert config.target_sr == 8000
        assert config.vad_aggressiveness == 1
        assert config.min_voiced_seconds == 0.5
    
    def test_config_validation_invalid_sr(self):
        """Test validation fails for invalid sample rate."""
        with pytest.raises(ConfigurationError, match="target_sr must be positive"):
            AudioNetConfig(target_sr=-1)
    
    def test_config_validation_invalid_vad_frame(self):
        """Test validation fails for invalid VAD frame size."""
        with pytest.raises(ConfigurationError, match="vad_frame_ms must be 10, 20, or 30"):
            AudioNetConfig(vad_frame_ms=25)
    
    def test_config_validation_invalid_vad_aggressiveness(self):
        """Test validation fails for invalid VAD aggressiveness."""
        with pytest.raises(ConfigurationError, match="vad_aggressiveness must be 0-3"):
            AudioNetConfig(vad_aggressiveness=5)
    
    def test_config_validation_invalid_voiced_ratio(self):
        """Test validation fails for invalid voiced ratio."""
        with pytest.raises(ConfigurationError, match="min_voiced_ratio must be between 0 and 1"):
            AudioNetConfig(min_voiced_ratio=1.5)
    
    def test_config_validation_invalid_loudness_range(self):
        """Test validation fails for invalid loudness range."""
        with pytest.raises(ConfigurationError, match="loudness_ok_min_dbfs must be <= loudness_ok_max_dbfs"):
            AudioNetConfig(
                loudness_ok_min_dbfs=-5.0,
                loudness_ok_max_dbfs=-10.0
            )
    
    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = AudioNetConfig(target_sr=8000, min_voiced_seconds=2.0)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['target_sr'] == 8000
        assert config_dict['min_voiced_seconds'] == 2.0

    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        config_dict = {
            'target_sr': 8000,
            'min_voiced_seconds': 2.0,
            'vad_aggressiveness': 3,
        }
        
        config = AudioNetConfig.from_dict(config_dict)
        
        assert config.target_sr == 8000
        assert config.min_voiced_seconds == 2.0
        assert config.vad_aggressiveness == 3

    def test_config_validation_multiple_errors(self):
        """Test that multiple validation errors are reported."""
        with pytest.raises(ConfigurationError) as exc_info:
            AudioNetConfig(
                target_sr=-1,
                vad_frame_ms=25,
                min_voiced_ratio=2.0
            )
        
        error_msg = str(exc_info.value)
        assert "target_sr must be positive" in error_msg
        assert "vad_frame_ms must be 10, 20, or 30" in error_msg
        assert "min_voiced_ratio must be between 0 and 1" in error_msg