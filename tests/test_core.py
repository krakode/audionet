"""Tests for AudioNet core functionality."""

import numpy as np
import pytest
import tempfile
import soundfile as sf
from unittest.mock import Mock, patch

from audionet import AudioNet, AudioNetConfig, AudioNetResult
from audionet.exceptions import (
    AudioLoadError,
    AudioValidationError,
    UnsupportedFormatError,
    VADError,
    FileSizeError,
    MalformedAudioError
)


class TestAudioNet:
    """Test AudioNet class."""
    
    @pytest.fixture
    def processor(self):
        """Create AudioNet processor for testing."""
        return AudioNet()
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create a simple sine wave with some noise
        y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        return y.astype(np.float32), sr
    
    @pytest.fixture
    def temp_audio_file(self, sample_audio):
        """Create temporary audio file."""
        y, sr = sample_audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, y, sr)
            yield f.name
    
    def test_init_default_config(self):
        """Test AudioNet initialization with default config."""
        processor = AudioNet()
        
        assert isinstance(processor.config, AudioNetConfig)
        assert processor.config.target_sr == 16000
        assert processor.progress_callback is None
        assert processor.logger is not None
    
    def test_init_custom_config(self):
        """Test AudioNet initialization with custom config."""
        config = AudioNetConfig(target_sr=8000, min_voiced_seconds=0.5)
        processor = AudioNet(config=config)
        
        assert processor.config.target_sr == 8000
        assert processor.config.min_voiced_seconds == 0.5
    
    def test_process_numpy_array_success(self, processor, sample_audio):
        """Test processing NumPy array successfully."""
        y, sr = sample_audio
        
        result = processor(y, sr=sr)
        
        assert isinstance(result, AudioNetResult)
        assert result.ok is True
        assert result.reason is None
        assert isinstance(result.y, np.ndarray)
        assert result.sr == processor.config.target_sr
        assert len(result.metrics) > 0
        assert result.processing_time is not None
    
    def test_process_file_path_success(self, processor, temp_audio_file):
        """Test processing file path successfully."""
        result = processor(temp_audio_file)
        
        assert result.ok is True
        assert isinstance(result.y, np.ndarray)
        assert result.sr == processor.config.target_sr
    
    def test_process_bytes_success(self, processor, sample_audio):
        """Test processing bytes successfully."""
        y, sr = sample_audio
        
        # Create bytes from audio
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            sf.write(f.name, y, sr)
            with open(f.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
        
        result = processor(audio_bytes)
        
        assert result.ok is True
        assert isinstance(result.y, np.ndarray)
    
    def test_process_nonexistent_file(self, processor):
        """Test processing nonexistent file raises error."""
        result = processor("nonexistent_file.wav")
        
        assert result.ok is False
        assert "not found" in result.reason.lower()
    
    def test_process_unsupported_format(self, processor):
        """Test processing unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.xyz') as f:
            result = processor(f.name)
            
            assert result.ok is False
            assert "unsupported" in result.reason.lower()
    
    def test_process_empty_audio(self, processor):
        """Test processing empty audio."""
        empty_audio = np.array([], dtype=np.float32)
        
        result = processor(empty_audio, sr=16000)
        
        assert result.ok is False
        assert "empty" in result.reason.lower()
    
    def test_process_too_quiet_audio(self, processor):
        """Test processing audio that's too quiet."""
        # Create very quiet audio
        sr = 16000
        y = 0.001 * np.random.randn(sr * 2)  # Very quiet 2-second audio
        
        result = processor(y, sr=sr)
        
        assert result.ok is False
        assert "too quiet" in result.reason.lower()
    
    def test_process_clipped_audio(self, processor):
        """Test processing heavily clipped audio."""
        # Create heavily clipped audio
        sr = 16000
        y = np.ones(sr * 2)  # Fully clipped 2-second audio
        y[::2] = -1  # Half samples at -1, half at +1
        
        result = processor(y, sr=sr)
        
        assert result.ok is False
        assert "clipping" in result.reason.lower()
    
    def test_process_dc_offset_audio(self, processor):
        """Test processing audio with excessive DC offset."""
        # Create audio with large DC offset
        sr = 16000
        y = 0.5 + 0.1 * np.random.randn(sr * 2)  # Large DC offset
        
        result = processor(y, sr=sr)
        
        assert result.ok is False
        assert "dc" in result.reason.lower()
    
    def test_process_no_speech_audio(self, processor):
        """Test processing audio with no detected speech."""
        # Create pure silence
        sr = 16000
        y = np.zeros(sr * 2)
        
        result = processor(y, sr=sr)
        
        assert result.ok is False
        assert "speech" in result.reason.lower()
    
    def test_process_with_progress_callback(self, sample_audio):
        """Test processing with progress callback."""
        y, sr = sample_audio
        progress_calls = []
        
        def progress_callback(message, progress_pct):
            progress_calls.append((message, progress_pct))
        
        config = AudioNetConfig(enable_progress_callback=True)
        processor = AudioNet(config=config, progress_callback=progress_callback)
        
        result = processor(y, sr=sr)
        
        assert result.ok is True
        assert len(progress_calls) > 0
        # Check that progress goes from 0 to 100
        assert any(call[1] == 0.0 for call in progress_calls)
        assert any(call[1] == 100.0 for call in progress_calls)
    
    def test_process_stereo_to_mono_conversion(self, processor):
        """Test conversion of stereo audio to mono."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create stereo audio
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])
        
        result = processor(stereo, sr=sr)
        
        assert result.ok is True
        assert result.y.ndim == 1  # Should be mono
    
    def test_process_resampling(self, sample_audio):
        """Test audio resampling to target sample rate."""
        y, _ = sample_audio
        original_sr = 44100
        
        config = AudioNetConfig(target_sr=16000)
        processor = AudioNet(config=config)
        
        result = processor(y, sr=original_sr)
        
        assert result.ok is True
        assert result.sr == 16000
        # Length should change due to resampling
        expected_length = int(len(y) * 16000 / original_sr)
        assert abs(len(result.y) - expected_length) < 100  # Allow some tolerance
    
    def test_process_with_custom_config(self, sample_audio):
        """Test processing with custom configuration."""
        y, sr = sample_audio
        
        config = AudioNetConfig(
            min_voiced_seconds=0.1,  # Very lenient
            min_voiced_ratio=0.01,   # Very lenient
            loudness_ok_min_dbfs=-50.0  # Very lenient
        )
        processor = AudioNet(config=config)
        
        result = processor(y, sr=sr)
        
        assert result.ok is True
    
    def test_process_with_spectral_checks(self, sample_audio):
        """Test processing with spectral quality checks enabled."""
        y, sr = sample_audio
        
        config = AudioNetConfig(
            enable_spectral_checks=True,
            min_snr_db=5.0  # Reasonable SNR threshold
        )
        processor = AudioNet(config=config)
        
        result = processor(y, sr=sr)
        
        assert result.ok is True
        assert 'snr_db' in result.metrics
        assert 'spectral_centroid_hz' in result.metrics
    
    def test_process_max_duration_limit(self, processor):
        """Test processing with maximum duration limit."""
        sr = 16000
        long_audio = np.random.randn(sr * 5)  # 5 seconds
        
        config = AudioNetConfig(max_duration_seconds=3.0)
        processor = AudioNet(config=config)
        
        result = processor(long_audio, sr=sr)
        
        assert result.ok is False
        assert "duration" in result.reason.lower()
    
    def test_rms_dbfs_calculation(self):
        """Test RMS dBFS calculation."""
        # Test with known signal
        y = np.array([0.5, -0.5, 0.5, -0.5])  # RMS = 0.5, dBFS = 20*log10(0.5) ≈ -6.02
        
        rms_dbfs = AudioNet._rms_dbfs(y)
        
        assert abs(rms_dbfs - (-6.02)) < 0.1
    
    def test_rms_dbfs_empty_array(self):
        """Test RMS dBFS calculation with empty array."""
        y = np.array([])
        
        rms_dbfs = AudioNet._rms_dbfs(y)
        
        assert rms_dbfs == -np.inf
    
    @patch('audionet.core._HAS_WEBRTC_VAD', False)
    def test_vad_fallback_energy_based(self, processor, sample_audio):
        """Test VAD fallback to energy-based method."""
        y, sr = sample_audio
        
        result = processor(y, sr=sr)
        
        # Should still work with energy-based VAD
        assert isinstance(result, AudioNetResult)
        assert 'voiced_ratio' in result.metrics
        assert 'voiced_seconds' in result.metrics
    
    def test_custom_processors(self, sample_audio):
        """Test custom processor integration."""
        y, sr = sample_audio
        
        def custom_processor(audio, sample_rate, existing_metrics):
            return {"custom_metric": 42.0}
        
        config = AudioNetConfig()
        config.custom_processors = {"test_processor": custom_processor}
        
        processor = AudioNet(config=config)
        result = processor(y, sr=sr)
        
        assert result.ok is True
        assert "test_processor_custom_metric" in result.metrics
        assert result.metrics["test_processor_custom_metric"] == 42.0
    
    def test_file_size_limit_validation(self, temp_audio_file):
        """Test file size limit validation."""
        # Set a very small file size limit
        config = AudioNetConfig(max_file_size_mb=0.001)  # 0.001 MB = 1 KB
        processor = AudioNet(config=config)
        
        result = processor(temp_audio_file)
        
        assert result.ok is False
        assert "file size" in result.reason.lower() or "exceeds maximum" in result.reason.lower()
    
    def test_file_size_limit_disabled(self, temp_audio_file):
        """Test that file size limit can be disabled."""
        config = AudioNetConfig(max_file_size_mb=None)  # Disabled
        processor = AudioNet(config=config)
        
        result = processor(temp_audio_file)
        
        # Should not fail due to file size (may fail for other reasons)
        if not result.ok:
            assert "file size" not in result.reason.lower()
            assert "exceeds maximum" not in result.reason.lower()
    
    def test_malformed_detection_disabled(self, sample_audio):
        """Test that malformed detection can be disabled."""
        y, sr = sample_audio
        
        config = AudioNetConfig(enable_malformed_detection=False)
        processor = AudioNet(config=config)
        
        result = processor(y, sr=sr)
        
        # Should process normally without malformed detection
        assert isinstance(result, AudioNetResult)
    
    def test_audio_integrity_validation_unrealistic_sample_rate(self, processor):
        """Test audio integrity validation with unrealistic sample rate."""
        # Create a mock file path
        import pathlib
        mock_path = pathlib.Path("test.wav")
        
        # This should raise an error for unrealistic sample rate
        with pytest.raises(MalformedAudioError):
            processor._validate_audio_integrity(
                np.array([0.1, 0.2, 0.3]), 
                sr=500000,  # Unrealistically high sample rate
                audio_path=mock_path
            )
    
    def test_audio_integrity_validation_normal_audio(self, processor, sample_audio):
        """Test audio integrity validation with normal audio."""
        y, sr = sample_audio
        
        # Create a temporary file to get a real path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, y, sr)
            audio_path = pathlib.Path(f.name)
            
            # This should not raise any exceptions
            try:
                processor._validate_audio_integrity(y, sr, audio_path)
            except Exception as e:
                pytest.fail(f"Audio integrity validation failed unexpectedly: {e}")
            finally:
                # Clean up
                audio_path.unlink(missing_ok=True)