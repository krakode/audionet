"""AudioNet custom exceptions."""


class AudioNetError(Exception):
    """Base exception for AudioNet library."""
    pass


class AudioLoadError(AudioNetError):
    """Raised when audio loading fails."""
    pass


class AudioValidationError(AudioNetError):
    """Raised when audio validation fails."""
    pass


class ConfigurationError(AudioNetError):
    """Raised when configuration is invalid."""
    pass


class UnsupportedFormatError(AudioLoadError):
    """Raised when audio format is not supported."""
    pass


class VADError(AudioNetError):
    """Raised when Voice Activity Detection fails."""
    pass


class FileSizeError(AudioValidationError):
    """Raised when file size exceeds limits."""
    pass


class MalformedAudioError(AudioLoadError):
    """Raised when audio file is malformed or corrupted."""
    pass