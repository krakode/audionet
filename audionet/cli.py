"""Command-line interface for AudioNet."""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import List, Optional

import soundfile as sf

from . import AudioNet, AudioNetConfig
from .exceptions import AudioNetError


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_config(config_path: Optional[str]) -> AudioNetConfig:
    """Load configuration from file."""
    if not config_path:
        return AudioNetConfig()

    config_file = pathlib.Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            if config_file.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                # Assume YAML
                import yaml

                config_dict = yaml.safe_load(f)

        return AudioNetConfig.from_dict(config_dict)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def find_audio_files(
    input_path: pathlib.Path, recursive: bool = False
) -> List[pathlib.Path]:
    """Find audio files in directory."""
    audio_extensions = {".wav", ".mp3"}

    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in audio_extensions else []

    pattern = "**/*" if recursive else "*"
    files = []

    for ext in audio_extensions:
        files.extend(input_path.glob(f"{pattern}{ext}"))
        files.extend(input_path.glob(f"{pattern}{ext.upper()}"))

    return sorted(files)


def process_single_file(
    input_path: pathlib.Path,
    output_path: Optional[pathlib.Path],
    processor: AudioNet,
    logger: logging.Logger,
    export_metrics: bool = False,
) -> bool:
    """Process a single audio file."""
    try:
        logger.info(f"Processing: {input_path}")

        result = processor(str(input_path))

        if result.ok:
            logger.info(
                f"✓ Success: {input_path.name} " f"({result.processing_time:.3f}s)"
            )

            # Save processed audio if output path specified
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), result.y, result.sr)
                logger.info(f"Saved to: {output_path}")

            # Export metrics if requested
            if export_metrics:
                metrics_path = (
                    output_path.with_suffix(".json")
                    if output_path
                    else input_path.with_suffix(".metrics.json")
                )
                with open(metrics_path, "w") as f:
                    json.dump(
                        {
                            "ok": result.ok,
                            "reason": result.reason,
                            "processing_time": result.processing_time,
                            **result.metrics,
                        },
                        f,
                        indent=2,
                    )
                logger.debug(f"Metrics exported to: {metrics_path}")

            return True
        else:
            logger.error(f"✗ Failed: {input_path.name} - {result.reason}")
            return False

    except Exception as e:
        logger.error(f"✗ Error processing {input_path.name}: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AudioNet: Production-ready audio preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audionet input.wav                                    # Validate audio quality
  audionet input.wav --output processed.wav            # Process and save
  audionet audio_dir/ --recursive --output out_dir/    # Batch process directory
  audionet input.wav --config config.json --verbose   # Use custom config
        """,
    )

    parser.add_argument("input", help="Input audio file or directory")

    parser.add_argument("--output", "-o", help="Output file or directory")

    parser.add_argument("--config", "-c", help="Configuration file (JSON or YAML)")

    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process directories recursively"
    )

    parser.add_argument(
        "--export-metrics",
        action="store_true",
        help="Export processing metrics to JSON files",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    parser.add_argument(
        "--progress", action="store_true", help="Show processing progress"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    try:
        # Load configuration
        config = load_config(args.config)
        if args.progress:
            config.enable_progress_callback = True

        # Setup input/output paths
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)

        output_path = pathlib.Path(args.output) if args.output else None

        # Find files to process
        files_to_process = find_audio_files(input_path, args.recursive)
        if not files_to_process:
            logger.error(f"No audio files found in: {input_path}")
            sys.exit(1)

        logger.info(f"Found {len(files_to_process)} audio file(s)")

        # Progress callback
        def progress_callback(message: str, progress_pct: float):
            if args.progress:
                print(f"\r{message} [{progress_pct:.1f}%]", end="", flush=True)

        # Initialize processor
        processor = AudioNet(
            config=config,
            progress_callback=progress_callback if args.progress else None,
            logger=logger,
        )

        # Process files
        start_time = time.time()
        successful = 0
        failed = 0

        for i, input_file in enumerate(files_to_process):
            if args.progress:
                file_progress = (i / len(files_to_process)) * 100
                print(
                    f"\rProcessing file {i+1}/{len(files_to_process)} [{file_progress:.1f}%]",
                    end="",
                    flush=True,
                )

            # Determine output path
            if output_path:
                if output_path.is_dir() or (
                    len(files_to_process) > 1 and not output_path.suffix
                ):
                    # Output directory
                    output_dir = output_path
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = (
                        output_dir / f"{input_file.stem}_processed{input_file.suffix}"
                    )
                else:
                    # Single output file
                    output_file = output_path
            else:
                output_file = None

            success = process_single_file(
                input_file, output_file, processor, logger, args.export_metrics
            )

            if success:
                successful += 1
            else:
                failed += 1

        if args.progress:
            print()  # New line after progress

        # Summary
        total_time = time.time() - start_time
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total time: {total_time:.2f}s")

        if failed > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
