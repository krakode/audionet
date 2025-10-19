"""Pytest configuration and fixtures for AudioNet tests."""

import pytest
import numpy as np
import tempfile
import pathlib
import soundfile as sf
from unittest.mock import Mock


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a sine wave with some noise to simulate speech-like audio
    frequency = 440  # A4 note
    y = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    
    # Add some amplitude modulation to make it more speech-like
    mod_freq = 5  # 5 Hz modulation
    envelope = 1 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
    y = y * envelope
    
    # Normalize to prevent clipping
    y = y / np.max(np.abs(y)) * 0.7
    
    return y.astype(np.float32), sr


@pytest.fixture
def quiet_audio():
    """Generate very quiet audio for testing failure cases."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    
    # Very quiet noise
    y = 0.001 * np.random.randn(n_samples)
    
    return y.astype(np.float32), sr


@pytest.fixture
def silence():
    """Generate silent audio."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    
    y = np.zeros(n_samples, dtype=np.float32)
    
    return y, sr


@pytest.fixture
def clipped_audio():
    """Generate clipped audio for testing failure cases."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    
    # Create heavily clipped audio
    y = np.ones(n_samples, dtype=np.float32)
    y[::2] = -1  # Alternate between +1 and -1
    
    return y, sr


@pytest.fixture
def temp_audio_file(sample_audio):
    """Create a temporary audio file."""
    y, sr = sample_audio
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, y, sr)
        yield pathlib.Path(f.name)
        
        # Cleanup
        try:
            pathlib.Path(f.name).unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback for testing."""
    return Mock()


@pytest.fixture
def stereo_audio():
    """Generate stereo audio for testing mono conversion."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create different signals for left and right channels
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    
    stereo = np.column_stack([left, right])
    
    return stereo.astype(np.float32), sr