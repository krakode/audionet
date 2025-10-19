"""Configuration classes for AudioNet."""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Callable, Self

from .exceptions import ConfigurationError


@dataclass
class AudioNetConfig:
    """Configuration for the AudioNet processor."""

    # Core processing parameters
    target_sr: int = 16000

    # VAD parameters
    vad_frame_ms: int = 30
    vad_aggressiveness: int = 2

    # Speech validation thresholds
    min_voiced_seconds: float = 1.0
    min_voiced_ratio: float = 0.15

    # Loudness validation (RMS dBFS on voiced frames)
    loudness_ok_min_dbfs: float = -35.0
    loudness_ok_max_dbfs: float = -8.0

    # Quality validation thresholds
    clip_threshold: float = 0.999
    max_clip_fraction: float = 0.01
    max_abs_dc: float = 0.02

    # Fallback energy VAD parameters
    fallback_energy_thresh_dbfs: float = -45.0
    fallback_min_zcr: float = 0.005

    # Spectral validation parameters
    enable_spectral_checks: bool = False
    min_snr_db: float = 0.0  # Only reject if SNR is below 0 dB

    # Speaker metadata extraction
    enable_speaker_metadata: bool = False

    # Processing options
    max_duration_seconds: float | None = None
    min_duration_seconds: float = 3.0
    max_file_size_mb: float | None = 100.0
    enable_malformed_detection: bool = True

    # Logging and monitoring
    enable_progress_callback: bool = False
    log_level: int = logging.INFO

    def __post_init__(self):
        """Validate configuration parameters."""
        self.validate()

    def validate(self):
        """Validate configuration parameters and raise ConfigurationError if invalid."""
        errors = []
        if self.target_sr <= 0:
            errors.append("target_sr must be positive")

        if self.vad_frame_ms not in (10, 20, 30):
            errors.append("vad_frame_ms must be 10, 20, or 30")

        if not 0 <= self.vad_aggressiveness <= 3:
            errors.append("vad_aggressiveness must be 0-3")

        # Threshold validation
        if self.min_voiced_seconds < 0:
            errors.append("min_voiced_seconds must be non-negative")

        if not 0 <= self.min_voiced_ratio <= 1:
            errors.append("min_voiced_ratio must be between 0 and 1")

        # Loudness validation
        if self.loudness_ok_min_dbfs > self.loudness_ok_max_dbfs:
            errors.append("loudness_ok_min_dbfs must be <= loudness_ok_max_dbfs")

        # Quality validation
        if not 0 <= self.clip_threshold <= 1:
            errors.append("clip_threshold must be between 0 and 1")

        if not 0 <= self.max_clip_fraction <= 1:
            errors.append("max_clip_fraction must be between 0 and 1")

        if self.max_abs_dc < 0:
            errors.append("max_abs_dc must be non-negative")

        # Duration limits
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            errors.append("max_duration_seconds must be positive when specified")

        if self.min_duration_seconds <= 0:
            errors.append("min_duration_seconds must be positive")

        # File size limits
        if self.max_file_size_mb is not None and self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive when specified")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def to_dict(self) -> dict[str, any]:
        """
        Convert the configuration dataclass to a dictionary.

        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> Self:
        """
        Create a configuration instance from a dictionary.
        """
        valid_field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {
            key: value for key, value in config_dict.items() if key in valid_field_names
        }
        return cls(**filtered_dict)
