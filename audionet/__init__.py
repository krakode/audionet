"""AudioNet: Production-ready audio preprocessing for speaker verification pipelines."""

from .core import AudioNet, AudioNetConfig, AudioNetResult
from .exceptions import (
    AudioNetError,
    AudioLoadError,
    AudioValidationError,
    ConfigurationError,
    UnsupportedFormatError,
    VADError,
    FileSizeError,
    MalformedAudioError,
)


__all__ = [
    "AudioNet",
    "AudioNetConfig", 
    "AudioNetResult",
    "AudioNetError",
    "AudioLoadError",
    "AudioValidationError",
    "ConfigurationError",
    "UnsupportedFormatError",
    "VADError",
    "FileSizeError",
    "MalformedAudioError",
]