import io
import logging
import math
import pathlib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from .config import AudioNetConfig
from .exceptions import (
    AudioLoadError,
    AudioNetError,
    AudioValidationError,
    FileSizeError,
    MalformedAudioError,
    UnsupportedFormatError,
    VADError,
)

# Optional: WebRTC VAD (better than energy gating). If not installed, we fall back automatically.
try:
    import webrtcvad

    _HAS_WEBRTC_VAD = True
except ImportError:
    _HAS_WEBRTC_VAD = False
    warnings.warn(
        "WebRTC VAD not available. Install with: pip install webrtcvad", UserWarning
    )


@dataclass
class AudioNetResult:
    """Result object containing the processed audio and quality metrics."""

    y: np.ndarray
    sr: int
    ok: bool
    reason: str | None
    metrics: dict[str, float]
    processing_time: float | None = None


class AudioNet:
    """
    Audio preprocessor for speaker verification pipelines.

    Processing steps:
    1) Load (file path, bytes, or array), downmix to mono, resample to target SR
    2) Basic signal checks: DC offset, clipping
    3) VAD (WebRTC if available else energy-based fallback)
    4) Speech presence checks (voiced seconds / ratio)
    5) Loudness validation on voiced frames (RMS dBFS)
    6) Optional spectral quality checks
    """

    # Constants
    _EPSILON = 1e-12  # Small value to avoid log(0) or division by zero

    # The WebRTC VAD library operates on 16-bit signed integer audio samples.
    # These constants represent the min/max values for that format and are used
    # to scale our normalized float audio from [-1.0, 1.0] to [-32768, 32767].
    _INT16_MIN_VALUE = np.iinfo(np.int16).min
    _INT16_MAX_VALUE = np.iinfo(np.int16).max

    def __init__(
            self,
            config: Optional[AudioNetConfig] = None,
            progress_callback: Optional[Callable[[str, float], None]] = None,
            logger: Optional[logging.Logger] = None,
    ):
        """Initialize the AudioNet processor.

        Args:
            config: AudioNet configuration object
            progress_callback: Optional callback for progress updates (message, progress_pct)
            logger: Optional logger instance
        """
        self.config = config or AudioNetConfig()
        self.progress_callback = progress_callback
        self.logger = logger or self._setup_logger()

        self.logger.info(f"AudioNet initialized with config: {self.config.to_dict()}")

        if not _HAS_WEBRTC_VAD:
            self.logger.warning("WebRTC VAD not available, using energy-based fallback")

    def __call__(
            self, audio: str | pathlib.Path | bytes | np.ndarray, sr: int | None = None
    ) -> AudioNetResult:
        """
        Process and validate an audio input.

        Args:
            audio: Audio input (file path, bytes, or NumPy array)
            sr: Sample rate if audio is a NumPy array

        Returns:
            AudioNetResult with processed audio and validation results

        Raises:
            AudioNetError: For processing errors
            AudioLoadError: For loading errors
            AudioValidationError: For validation errors
        """
        import time

        start_time = time.time()
        y, sr = np.array([]), self.config.target_sr
        metrics = {}

        try:
            self._report_progress("Loading audio", 0.0)
            y, sr, metrics = self._load_and_resample(audio, sr_override=sr)

            self._report_progress("Analyzing audio quality", 20.0)

            # Early validation checks
            self._validate_basic_quality(metrics)
            self._validate_silence_quality(metrics)

            self._report_progress("Running voice activity detection", 40.0)
            voiced_mask, frame_len = self._run_vad(y, sr)

            # Add VAD metrics
            metrics.update(self._compute_vad_metrics(voiced_mask, frame_len, sr))

            # Validate speech presence
            self._validate_speech_presence(metrics)

            self._report_progress("Analyzing voiced segments", 70.0)
            y_voiced = self._extract_voiced_signal(y, voiced_mask, frame_len)

            # Compute loudness on voiced frames
            if y_voiced.size > 0:
                metrics["rms_dbfs_voiced"] = self._rms_dbfs(y_voiced)
            else:
                metrics["rms_dbfs_voiced"] = -np.inf

            # Validate loudness
            self._validate_loudness(metrics["rms_dbfs_voiced"])

            # Optional spectral quality checks
            if self.config.enable_spectral_checks:
                spectral_metrics = self._compute_spectral_metrics(y_voiced, self.config.target_sr)
                metrics.update(spectral_metrics)

            # Compute harmonic-to-noise ratio on voiced frames
            if self.config.enable_spectral_checks and y_voiced.size > 0:
                metrics["hnr_db"] = self._estimate_hnr(y_voiced, sr)
            elif self.config.enable_spectral_checks:
                metrics["hnr_db"] = 0.0  # Default if no voiced audio

            # Optional speaker metadata extraction
            if self.config.enable_speaker_metadata:
                self._report_progress("Extracting speaker metadata", 95.0)
                speaker_metrics = self._compute_speaker_metadata(
                    y_voiced, y, sr, voiced_mask, frame_len
                )
                metrics.update(speaker_metrics)

            if self.config.enable_spectral_checks:
                self._report_progress("Running spectral validation", 98.0)
                self._validate_spectral_quality(metrics)

            processing_time = time.time() - start_time
            self._report_progress("Processing complete", 100.0)

            self.logger.info(f"Audio processed successfully in {processing_time:.3f}s")

            return AudioNetResult(
                y=y,
                sr=sr,
                ok=True,
                reason=None,
                metrics=metrics,
                processing_time=processing_time,
            )

        except (
                AudioLoadError,
                AudioValidationError,
                VADError,
                FileSizeError,
                MalformedAudioError,
        ) as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Audio processing failed: {e}")
            return self._create_failure_result(y, sr, str(e), metrics, processing_time)
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Unexpected error during processing: {e}")
            raise AudioNetError(f"Unexpected processing error: {e}") from e

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for AudioNet."""
        logger = logging.getLogger(__name__)
        logger.setLevel(self.config.log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _report_progress(self, message: str, progress_pct: float):
        """Report progress if callback is enabled."""
        if self.config.enable_progress_callback and self.progress_callback:
            self.progress_callback(message, progress_pct)

    def _load_and_resample(
            self,
            audio: Union[str, pathlib.Path, bytes, np.ndarray],
            sr_override: Optional[int],
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        """Load audio from various sources and resample to target SR."""
        try:
            if isinstance(audio, np.ndarray):
                y = audio.astype(np.float32, copy=False)
                sr = sr_override if sr_override else self.config.target_sr
                if sr_override is None:
                    self.logger.warning(
                        "No sample rate provided for NumPy array, assuming target SR"
                    )

            elif isinstance(audio, (str, pathlib.Path, bytes)):
                if isinstance(audio, bytes):
                    # Check bytes size
                    if self.config.max_file_size_mb is not None:
                        file_size_mb = len(audio) / (1024 * 1024)
                        if file_size_mb > self.config.max_file_size_mb:
                            raise FileSizeError(
                                f"Audio data size ({file_size_mb:.1f} MB) exceeds maximum "
                                f"({self.config.max_file_size_mb} MB)"
                            )

                    try:
                        f = io.BytesIO(audio)
                        y, sr = sf.read(f, always_2d=False, dtype="float32")

                        # Basic malformed audio detection for bytes
                        if self.config.enable_malformed_detection:
                            self._validate_audio_integrity_bytes(y, sr, len(audio))

                    except sf.LibsndfileError as e:
                        if self.config.enable_malformed_detection:
                            raise MalformedAudioError(
                                f"Audio data appears to be corrupted or malformed: {e}"
                            ) from e
                        else:
                            raise AudioLoadError(
                                f"Failed to read audio data: {e}"
                            ) from e
                else:
                    audio_path = pathlib.Path(audio)
                    if not audio_path.exists():
                        raise AudioLoadError(f"Audio file not found: {audio_path}")

                    # Check file size
                    if self.config.max_file_size_mb is not None:
                        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                        if file_size_mb > self.config.max_file_size_mb:
                            raise FileSizeError(
                                f"File size ({file_size_mb:.1f} MB) exceeds maximum "
                                f"({self.config.max_file_size_mb} MB)"
                            )

                    # Check file extension
                    supported_extensions = {".wav", ".mp3"}
                    if audio_path.suffix.lower() not in supported_extensions:
                        raise UnsupportedFormatError(
                            f"Unsupported audio format: {audio_path.suffix}. "
                            f"Supported: {supported_extensions}"
                        )

                    # Attempt to read the file with malformed detection
                    try:
                        y, sr = sf.read(
                            str(audio_path), always_2d=False, dtype="float32"
                        )

                        # Basic malformed audio detection
                        if self.config.enable_malformed_detection:
                            self._validate_audio_integrity(y, sr, audio_path)

                    except sf.LibsndfileError as e:
                        if self.config.enable_malformed_detection:
                            raise MalformedAudioError(
                                f"Audio file appears to be corrupted or malformed: {e}"
                            ) from e
                        else:
                            raise AudioLoadError(
                                f"Failed to read audio file: {e}"
                            ) from e
            else:
                raise AudioLoadError(f"Unsupported audio input type: {type(audio)}")

            # Check for empty audio
            if y.size == 0:
                raise AudioLoadError("Audio file is empty")

            # Convert to mono if stereo/multichannel
            if y.ndim == 2:
                y = np.mean(y, axis=1)
                self.logger.debug("Converted multichannel audio to mono")

            # Remove NaNs/Infs
            if np.any(~np.isfinite(y)):
                self.logger.warning("Removing NaN/Inf values from audio")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Duration checks
            duration = len(y) / sr
            if duration < self.config.min_duration_seconds:
                raise AudioValidationError(
                    f"Audio too short ({duration:.2f}s) for meaningful feature extraction "
                    f"(minimum: {self.config.min_duration_seconds}s)"
                )

            if (
                    self.config.max_duration_seconds
                    and duration > self.config.max_duration_seconds
            ):
                raise AudioValidationError(
                    f"Audio duration ({duration:.2f}s) exceeds maximum "
                    f"({self.config.max_duration_seconds}s)"
                )

            # Resample to target SR
            if sr != self.config.target_sr:
                self.logger.debug(f"Resampling from {sr} to {self.config.target_sr} Hz")
                y = librosa.resample(
                    y,
                    orig_sr=sr,
                    target_sr=self.config.target_sr,
                    res_type="kaiser_best",
                )
                sr = self.config.target_sr

            # Calculate pre-normalization metrics
            pre_norm_metrics = self._compute_pre_normalization_metrics(y)
            pre_norm_metrics["duration"] = float(duration)

            # Normalization
            peak = np.max(np.abs(y)) + self._EPSILON
            if peak > 0:
                y = y / peak

            return y.astype(np.float32, copy=False), sr, pre_norm_metrics

        except sf.LibsndfileError as e:
            raise AudioLoadError(f"Failed to load audio: {e}") from e
        except Exception as e:
            if isinstance(
                    e, (AudioLoadError, UnsupportedFormatError, AudioValidationError)
            ):
                raise
            raise AudioLoadError(f"Unexpected error loading audio: {e}") from e

    def _compute_pre_normalization_metrics(self, y: np.ndarray) -> dict[str, float]:
        """Compute metrics that must be calculated before normalization."""
        metrics = {}

        # DC offset & clipping
        metrics["dc_offset"] = float(np.mean(y))
        metrics["clip_fraction"] = float(
            np.mean(np.abs(y) >= self.config.clip_threshold)
        )
        metrics["rms_dbfs_full"] = self._rms_dbfs(y)
        metrics["peak_amplitude"] = float(np.max(np.abs(y)))

        # Silence detection metrics
        silence_metrics = self._compute_silence_metrics(y)
        metrics.update(silence_metrics)

        return metrics

    def _validate_basic_quality(self, metrics: dict[str, float]):
        """Validate basic signal quality."""
        if abs(metrics["dc_offset"]) > self.config.max_abs_dc:
            raise AudioValidationError(
                f"Excessive DC offset: {metrics['dc_offset']:.4f} "
                f"(max: {self.config.max_abs_dc})"
            )

        if metrics["clip_fraction"] > self.config.max_clip_fraction:
            raise AudioValidationError(
                f"Excessive clipping: {metrics['clip_fraction']:.3f} "
                f"(max: {self.config.max_clip_fraction})"
            )

    def _run_vad(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Run Voice Activity Detection on the audio signal."""
        try:
            frame_len = int(sr * self.config.vad_frame_ms / 1000)
            n_frames = len(y) // frame_len

            if n_frames == 0:
                return np.zeros(0, dtype=bool), frame_len

            frames = y[: n_frames * frame_len].reshape(n_frames, frame_len)

            # Try WebRTC VAD first
            if (
                    _HAS_WEBRTC_VAD
                    and sr in (8000, 16000, 32000, 48000)
                    and self.config.vad_frame_ms in (10, 20, 30)
            ):

                try:
                    vad = webrtcvad.Vad(self.config.vad_aggressiveness)
                    # WebRTC expects 16-bit PCM
                    pcm16 = (
                        (frames * self._INT16_MAX_VALUE)
                        .clip(self._INT16_MIN_VALUE, self._INT16_MAX_VALUE)
                        .astype(np.int16)
                    )

                    voiced = [vad.is_speech(f.tobytes(), sr) for f in pcm16]
                    self.logger.debug("Used WebRTC VAD for voice activity detection")
                    return np.array(voiced, dtype=bool), frame_len

                except Exception as e:
                    self.logger.warning(
                        f"WebRTC VAD failed, falling back to energy-based: {e}"
                    )

            # Fallback: energy + simple ZCR guard
            rms = np.sqrt(np.mean(frames ** 2, axis=1)) + self._EPSILON
            dbfs = 20.0 * np.log10(rms)
            # Zero-crossing rate to avoid passing steady tones
            zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2.0

            voiced = (dbfs > self.config.fallback_energy_thresh_dbfs) & (
                    zcr > self.config.fallback_min_zcr
            )

            self.logger.debug("Used energy-based VAD for voice activity detection")
            return voiced.astype(bool), frame_len

        except Exception as e:
            raise VADError(f"Voice Activity Detection failed: {e}") from e

    def _compute_vad_metrics(
            self, voiced_mask: np.ndarray, frame_len: int, sr: int
    ) -> dict[str, float]:
        """Compute VAD-related metrics."""
        if len(voiced_mask) == 0:
            return {"voiced_ratio": 0.0, "voiced_seconds": 0.0}

        return {
            "voiced_ratio": float(np.mean(voiced_mask)),
            "voiced_seconds": float(np.sum(voiced_mask)) * (frame_len / sr),
        }

    def _validate_speech_presence(self, metrics: dict[str, float]):
        """Validate that sufficient speech is present."""
        if (
                metrics["voiced_seconds"] < self.config.min_voiced_seconds
                and metrics["voiced_ratio"] < self.config.min_voiced_ratio
        ):
            raise AudioValidationError(
                f"Insufficient speech detected: {metrics['voiced_seconds']:.2f}s "
                f"({metrics['voiced_ratio']:.1%} ratio). Required: "
                f"{self.config.min_voiced_seconds}s or {self.config.min_voiced_ratio:.1%} ratio"
            )

    def _extract_voiced_signal(
            self, y: np.ndarray, voiced_mask: np.ndarray, frame_len: int
    ) -> np.ndarray:
        """Extract signal from voiced frames only."""
        if voiced_mask.size == 0 or not np.any(voiced_mask):
            return np.array([], dtype=y.dtype)

        n_frames = len(y) // frame_len
        y_truncated = y[: n_frames * frame_len]
        frames = y_truncated.reshape(n_frames, frame_len)
        kept_frames = frames[voiced_mask]

        return kept_frames.reshape(-1)

    def _validate_loudness(self, rms_dbfs_voiced: float):
        """Validate loudness on voiced segments."""
        if rms_dbfs_voiced < self.config.loudness_ok_min_dbfs:
            raise AudioValidationError(
                f"Audio too quiet: {rms_dbfs_voiced:.1f} dBFS "
                f"(min: {self.config.loudness_ok_min_dbfs} dBFS)"
            )

        if rms_dbfs_voiced > self.config.loudness_ok_max_dbfs:
            raise AudioValidationError(
                f"Audio too loud or clipping risk: {rms_dbfs_voiced:.1f} dBFS "
                f"(max: {self.config.loudness_ok_max_dbfs} dBFS)"
            )

    def _compute_spectral_metrics(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Compute spectral quality metrics."""
        try:
            if y.size == 0:
                return {
                    "snr_db": 0.0,
                    "hnr_db": 0.0,
                    "spectral_rolloff_hz": 0.0,
                    "spectral_centroid_hz": 0.0,
                    "zero_crossing_rate": 0.0,
                    "spectral_flatness": 1.0,
                }

            # STFT magnitude
            stft = librosa.stft(y, hop_length=512)
            magnitude = np.abs(stft)

            # Rough SNR: mean vs 10th percentile
            noise_floor = np.percentile(magnitude, 10)
            signal_level = np.mean(magnitude)
            snr_db = 20 * np.log10((signal_level + self._EPSILON) / (noise_floor + self._EPSILON))
            hnr_db = self._estimate_hnr(y, sr)

            # ZCR
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

            # Rolloff + centroid
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y, n_fft=512)[0]

            return {
                "snr_db": float(snr_db),
                "hnr_db": float(hnr_db),
                "spectral_centroid_hz": float(np.mean(centroid)),
                "spectral_rolloff_hz": float(np.mean(rolloff)),
                "zero_crossing_rate": zcr,
                "spectral_flatness": float(np.mean(flatness)),
            }
        except Exception as e:
            self.logger.warning(f"Spectral analysis failed: {e}")
            return {
                "snr_db": 0.0,
                "spectral_centroid_hz": 0.0,
                "spectral_rolloff_hz": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_flatness": 1.0,
            }

    @staticmethod
    def _rms_dbfs(x: np.ndarray) -> float:
        """Calculate RMS of signal in dBFS."""
        if x.size == 0:
            return -np.inf
        rms = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + AudioNet._EPSILON)
        return 20.0 * math.log10(rms)

    def _validate_audio_integrity(
            self, y: np.ndarray, sr: int, audio_path: pathlib.Path
    ):
        """Perform basic integrity checks on loaded audio data."""
        # Check for reasonable sample rate
        if sr <= 0 or sr > 192000:
            raise MalformedAudioError(
                f"Unrealistic sample rate ({sr} Hz) in {audio_path.name}. "
                "Audio file may be corrupted."
            )

        # Check for reasonable audio length vs file size
        expected_samples = (
                audio_path.stat().st_size // 4
        )  # Rough estimate for 32-bit float
        actual_samples = len(y)

        if actual_samples < expected_samples * 0.1:  # Less than 10% of expected samples
            self.logger.warning(
                f"Audio length ({actual_samples} samples) much shorter than expected "
                f"from file size in {audio_path.name}. File may be corrupted."
            )

        # Check for suspicious patterns that indicate corruption
        if len(y) > 1000:  # Only check if we have enough samples
            # Check for excessive identical consecutive samples (could indicate corruption)
            consecutive_identical = 0
            max_consecutive = 0

            for i in range(1, min(len(y), 10000)):  # Check first 10k samples max
                # Use np.all() to handle multi-channel audio correctly
                if np.all(abs(y[i] - y[i - 1]) < 1e-8):  # Essentially identical
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 0

            # If more than 1000 consecutive identical samples, flag as suspicious
            if max_consecutive > 1000:
                self.logger.warning(
                    f"Detected {max_consecutive} consecutive identical samples in {audio_path.name}. "
                    "This may indicate audio corruption or encoding issues."
                )

    def _validate_audio_integrity_bytes(self, y: np.ndarray, sr: int, byte_length: int):
        """Perform basic integrity checks on loaded audio data from bytes."""
        # Check for reasonable sample rate
        if sr <= 0 or sr > 192000:
            raise MalformedAudioError(
                f"Unrealistic sample rate ({sr} Hz) in audio data. "
                "Audio data may be corrupted."
            )

        # Check for reasonable audio length vs byte size
        expected_samples = byte_length // 4  # Rough estimate for 32-bit float
        actual_samples = len(y)

        if actual_samples < expected_samples * 0.1:  # Less than 10% of expected samples
            self.logger.warning(
                f"Audio length ({actual_samples} samples) much shorter than expected "
                f"from data size. Audio data may be corrupted."
            )

        # Check for suspicious patterns
        if len(y) > 1000:
            consecutive_identical = 0
            max_consecutive = 0

            for i in range(1, min(len(y), 10000)):  # Check first 10k samples max
                if np.all(abs(y[i] - y[i - 1]) < 1e-8):  # Essentially identical
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 0

            if max_consecutive > 1000:
                self.logger.warning(
                    f"Detected {max_consecutive} consecutive identical samples in audio data. "
                    "This may indicate audio corruption or encoding issues."
                )

    def _create_failure_result(
            self,
            y: np.ndarray,
            sr: int,
            reason: str,
            metrics: dict[str, float],
            processing_time: float,
    ) -> AudioNetResult:
        """Create a failure result object."""
        return AudioNetResult(
            y=y,
            sr=sr,
            ok=False,
            reason=reason,
            metrics=metrics,
            processing_time=processing_time,
        )

    def _compute_silence_metrics(self, y: np.ndarray) -> dict[str, float]:
        """Compute silence-related metrics for validation."""
        if y.size == 0:
            return {"digital_silence_ratio": 1.0, "excessive_silence_ratio": 1.0}

        # Digital silence detection (pure zeros or near-zeros)
        digital_silence_mask = np.abs(y) < 1e-6
        digital_silence_ratio = float(np.mean(digital_silence_mask))

        # Excessive silence detection using RMS in small frames
        frame_length = int(0.025 * self.config.target_sr)  # 25ms frames
        if frame_length == 0:
            frame_length = 1

        n_frames = len(y) // frame_length
        if n_frames == 0:
            return {
                "digital_silence_ratio": digital_silence_ratio,
                "excessive_silence_ratio": 1.0,
            }

        # Calculate RMS for each frame
        y_frames = y[: n_frames * frame_length].reshape(n_frames, frame_length)
        frame_rms = np.sqrt(np.mean(y_frames ** 2, axis=1)) + self._EPSILON
        frame_dbfs = 20.0 * np.log10(frame_rms)

        # Count frames below silence threshold
        silence_frames = frame_dbfs < -60.0  # -60 dBFS threshold
        excessive_silence_ratio = float(np.mean(silence_frames))

        return {
            "digital_silence_ratio": digital_silence_ratio,
            "excessive_silence_ratio": excessive_silence_ratio,
        }

    def _validate_silence_quality(self, metrics: dict[str, float]):
        """Validate silence-related quality metrics."""
        # Check for completely digital silence (pure zeros)
        if metrics.get("digital_silence_ratio", 0) > 0.95:
            raise AudioValidationError(
                f"Audio is mostly digital silence ({metrics['digital_silence_ratio']:.1%}). "
                "File may be corrupted or empty."
            )

        # Check for excessive silence
        if metrics.get("excessive_silence_ratio", 0) > 0.8:
            raise AudioValidationError(
                f"Excessive silence detected ({metrics['excessive_silence_ratio']:.1%}). "
                "Audio may not contain sufficient speech content for feature extraction."
            )

    def _validate_spectral_quality(self, metrics: dict[str, float]):
        """Lean spectral rejection: only block extreme, obviously bad cases."""
        if not self.config.enable_spectral_checks:
            return

        # SNR sanity check
        snr_db = metrics.get("snr_db", 0.0)
        if snr_db < 0:
            raise AudioValidationError(
                f"Signal-to-noise ratio too low ({snr_db:.1f} dB). Audio likely corrupted or pure noise."
            )

        # Spectral flatness → 1.0 is noise-like, near 0 is tonal
        flatness = metrics.get("spectral_flatness", 0.0)
        if flatness > 0.95:
            raise AudioValidationError(
                f"Spectral flatness too high ({flatness:.2f}). Audio dominated by noise, unlikely to contain speech."
            )

        # Very extreme rolloff values only
        rolloff = metrics.get("spectral_rolloff_hz")
        if rolloff is not None:
            if rolloff < 300:
                raise AudioValidationError(
                    f"Spectral rolloff too low ({rolloff:.0f} Hz). Audio may be hum/static."
                )
            if rolloff > 7000:
                raise AudioValidationError(
                    f"Spectral rolloff too high ({rolloff:.0f} Hz). Audio may contain only noise."
                )

    def _compute_speaker_metadata(
            self,
            y_voiced: np.ndarray,
            y_full: np.ndarray,
            sr: int,
            voiced_mask: np.ndarray,
            frame_len: int,
    ) -> Dict[str, float]:
        """
        Compute speaker-specific metadata for pattern analysis in speaker verification.

        Args:
            y_voiced: Voiced segments only
            y_full: Full audio signal
            sr: Sample rate
            voiced_mask: Boolean mask indicating voiced frames
            frame_len: Frame length used for VAD

        Returns:
            Dictionary of speaker metadata features
        """
        metadata = {}

        try:
            if y_voiced.size == 0:
                # Return default values if no voiced audio
                return {
                    "f0_mean": 0.0,
                    "f0_std": 0.0,
                    "f0_range": 0.0,
                    "f0_median": 0.0,
                    "pitch_stability": 0.0,
                    "voiced_unvoiced_ratio": 0.0,
                    "average_pause_duration": 0.0,
                    "pause_count": 0,
                    "speech_segment_count": 0,
                    "average_speech_segment_duration": 0.0,
                }

            # F0 (fundamental frequency) analysis
            f0_metadata = self._compute_f0_features(y_voiced, sr)
            metadata.update(f0_metadata)

            # Temporal speech patterns
            temporal_metadata = self._compute_temporal_patterns(
                voiced_mask, frame_len, sr
            )
            metadata.update(temporal_metadata)


        except Exception as e:
            self.logger.warning(f"Speaker metadata computation failed: {e}")
            # Return zeros for all expected features
            metadata = {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_range": 0.0,
                "f0_median": 0.0,
                "pitch_stability": 0.0,
                "voiced_unvoiced_ratio": 0.0,
                "average_pause_duration": 0.0,
                "pause_count": 0,
                "speech_segment_count": 0,
                "average_speech_segment_duration": 0.0,
                "hnr_db": 0.0,
            }

        return metadata

    def _compute_f0_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        """Compute F0 (fundamental frequency) related features."""
        try:
            # Use librosa's piptrack for F0 estimation
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, threshold=0.1, fmin=50, fmax=400
            )

            # 1. Get the index of the max magnitude for each time frame
            indices = np.argmax(magnitudes, axis=0)
            # 2. Use advanced indexing to get the pitch at those indices
            # np.arange creates the column indices [0, 1, 2, ...]
            f0_array = pitches[indices, np.arange(pitches.shape[1])]
            # 3. Filter out unvoiced frames where pitch is 0
            f0_array = f0_array[f0_array > 0]

            if f0_array.size == 0:
                return {
                    "f0_mean": 0.0,
                    "f0_std": 0.0,
                    "f0_range": 0.0,
                    "f0_median": 0.0,
                    "pitch_stability": 0.0,
                }

            # Basic F0 statistics
            f0_mean = float(np.mean(f0_array))
            f0_std = float(np.std(f0_array))
            f0_range = float(np.max(f0_array) - np.min(f0_array))
            f0_median = float(np.median(f0_array))

            # Pitch stability (inverse of coefficient of variation)
            pitch_stability = float(1.0 / (1.0 + f0_std / (f0_mean + self._EPSILON)))

            return {
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_range": f0_range,
                "f0_median": f0_median,
                "pitch_stability": pitch_stability,
            }

        except Exception as e:
            self.logger.warning(f"F0 feature computation failed: {e}")
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_range": 0.0,
                "f0_median": 0.0,
                "pitch_stability": 0.0,
            }

    def _compute_temporal_patterns(
            self, voiced_mask: np.ndarray, frame_len: int, sr: int
    ) -> dict[str, float]:
        """Compute temporal speech patterns (pauses, segments, etc.)."""
        try:
            if len(voiced_mask) == 0:
                return {
                    "voiced_unvoiced_ratio": 0.0,
                    "average_pause_duration": 0.0,
                    "pause_count": 0,
                    "speech_segment_count": 0,
                    "average_speech_segment_duration": 0.0,
                }

            # Voiced/unvoiced ratio
            voiced_ratio = float(np.mean(voiced_mask))
            voiced_unvoiced_ratio = voiced_ratio / (1.0 - voiced_ratio + self._EPSILON)

            # Find speech and pause segments
            # Convert boolean mask to transitions
            transitions = np.diff(voiced_mask.astype(int))
            speech_starts = np.where(transitions == 1)[0] + 1
            speech_ends = np.where(transitions == -1)[0] + 1

            # Handle edge cases
            if voiced_mask[0]:
                speech_starts = np.concatenate([[0], speech_starts])
            if voiced_mask[-1]:
                speech_ends = np.concatenate([speech_ends, [len(voiced_mask)]])

            # Ensure equal number of starts and ends
            min_len = min(len(speech_starts), len(speech_ends))
            speech_starts = speech_starts[:min_len]
            speech_ends = speech_ends[:min_len]

            # Calculate speech segment durations
            speech_segments = speech_ends - speech_starts
            speech_segment_durations = speech_segments * (frame_len / sr)

            speech_segment_count = len(speech_segments)
            average_speech_segment_duration = (
                float(np.mean(speech_segment_durations))
                if len(speech_segment_durations) > 0
                else 0.0
            )

            # Calculate pause durations (gaps between speech segments)
            pause_durations = []
            if len(speech_ends) > 1:
                for i in range(len(speech_ends) - 1):
                    pause_length = speech_starts[i + 1] - speech_ends[i]
                    if pause_length > 0:
                        pause_durations.append(pause_length * (frame_len / sr))

            pause_count = len(pause_durations)
            average_pause_duration = (
                float(np.mean(pause_durations)) if pause_durations else 0.0
            )

            return {
                "voiced_unvoiced_ratio": voiced_unvoiced_ratio,
                "average_pause_duration": average_pause_duration,
                "pause_count": pause_count,
                "speech_segment_count": speech_segment_count,
                "average_speech_segment_duration": average_speech_segment_duration,
            }

        except Exception as e:
            self.logger.warning(f"Temporal pattern computation failed: {e}")
            return {
                "voiced_unvoiced_ratio": 0.0,
                "average_pause_duration": 0.0,
                "pause_count": 0,
                "speech_segment_count": 0,
                "average_speech_segment_duration": 0.0,
            }

    def _estimate_hnr(self, y: np.ndarray, sr: int) -> float:
        """Estimate Harmonic-to-Noise Ratio in dB."""
        try:
            # Simple HNR estimation using autocorrelation
            # Window the signal
            window_length = min(int(0.04 * sr), len(y))  # 40ms window
            if window_length < 100:
                return 0.0

            windowed = y[:window_length] * np.hanning(window_length)

            # Autocorrelation
            autocorr = np.correlate(windowed, windowed, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]

            # Find the maximum correlation (excluding lag 0)
            if len(autocorr) < 20:
                return 0.0

            max_corr = np.max(autocorr[20:])  # Skip first 20 samples

            # HNR approximation
            hnr = max_corr / (autocorr[0] - max_corr + self._EPSILON)
            hnr_db = 10 * math.log10(hnr + self._EPSILON)

            return float(np.clip(hnr_db, -20, 40))  # Reasonable range

        except Exception:
            return 0.0
