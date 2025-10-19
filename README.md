# AudioNet


AudioNet is a library for audio preprocessing specifically designed for speaker verification and machine learning pipelines. It provides comprehensive audio quality validation, voice activity detection, and preprocessing capabilities.

## Features

- **Multi-format Audio Loading**: Support for file paths, byte streams, and NumPy arrays
- **Voice Activity Detection**: WebRTC VAD with energy-based fallback
- **Quality Validation**: DC offset, clipping, loudness, and spectral quality checks
- **Speaker Metadata Extraction**: F0 analysis, temporal patterns, and voice quality metrics for speaker verification
- **Preprocessing Pipeline**: Mono conversion, resampling, normalization

## Quick Start

### Installation

```bash
# Basic installation
pip install audionet

# With WebRTC VAD support (recommended)
pip install audionet[vad]
```

### Basic Usage

```python
from audionet import AudioNet

# Initialize the processor
processor = AudioNet()

# Process audio file
result = processor("path/to/audio.wav")

if result.ok:
    print(f"Audio processed successfully")
    print(f"Shape: {result.y.shape}, Sample rate: {result.sr}")
    print(f"Metrics: {result.metrics}")
else:
    print(f"Processing failed: {result.reason}")
```

### CLI Usage

```bash
# Process a single file
audionet input.wav --output processed.wav

# Batch process directory
audionet input_dir/ --output output_dir/ --recursive

# Custom configuration
audionet input.wav --config config.yaml --verbose
```

## Configuration

AudioNet is configurable through the `AudioNetConfig` class:

```python
from audionet import AudioNet, AudioNetConfig

config = AudioNetConfig(
    target_sr=16000,
    vad_aggressiveness=2,
    min_voiced_seconds=1.0,
    loudness_ok_min_dbfs=-35.0,
    loudness_ok_max_dbfs=-8.0
)

processor = AudioNet(config)
```

### Speaker Metadata Extraction

AudioNet can extract speaker-specific metadata for pattern analysis in speaker verification systems:

```python
from audionet import AudioNet, AudioNetConfig

# Enable speaker metadata extraction
config = AudioNetConfig(
    target_sr=16000,
    enable_speaker_metadata=True
)

processor = AudioNet(config)
result = processor("path/to/audio.wav")

if result.ok:
    # Access speaker metadata
    print(f"F0 mean: {result.metrics['f0_mean']:.1f} Hz")
    print(f"Pitch stability: {result.metrics['pitch_stability']:.3f}")
    print(f"Voice quality (HNR): {result.metrics['hnr_db']:.1f} dB")
    print(f"Pause count: {result.metrics['pause_count']}")
    print(f"Average pause duration: {result.metrics['average_pause_duration']:.3f}s")
```

**Available Speaker Metadata:**
- **F0 Analysis**: `f0_mean`, `f0_std`, `f0_range`, `f0_median`, `pitch_stability`
- **Temporal Patterns**: `voiced_unvoiced_ratio`, `average_pause_duration`, `pause_count`, `speech_segment_count`, `average_speech_segment_duration`
- **Voice Quality**: `hnr_db` (Harmonic-to-Noise Ratio)

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SoundFile >= 0.10.0
- Librosa >= 0.9.0

### Optional Dependencies

- WebRTC VAD >= 2.0.0 (for enhanced voice activity detection)
