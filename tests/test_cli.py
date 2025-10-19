"""Tests for AudioNet CLI."""

import json
import tempfile
import pathlib
from unittest.mock import patch, Mock
import pytest
import numpy as np
import soundfile as sf

from audionet.cli import (
    load_config,
    find_audio_files,
    process_single_file,
    main
)
from audionet import AudioNet, AudioNetConfig


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, y, sr)
            yield pathlib.Path(f.name)
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config_data = {
            "target_sr": 8000,
            "min_voiced_seconds": 0.5,
            "vad_aggressiveness": 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            yield pathlib.Path(f.name)
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config(None)
        
        assert isinstance(config, AudioNetConfig)
        assert config.target_sr == 16000
    
    def test_load_config_from_json(self, temp_config_file):
        """Test loading configuration from JSON file."""
        config = load_config(str(temp_config_file))
        
        assert config.target_sr == 8000
        assert config.min_voiced_seconds == 0.5
        assert config.vad_aggressiveness == 1
    
    def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")
    
    def test_find_audio_files_single_file(self, sample_audio_file):
        """Test finding single audio file."""
        files = find_audio_files(sample_audio_file)
        
        assert len(files) == 1
        assert files[0] == sample_audio_file
    
    def test_find_audio_files_directory(self, sample_audio_file):
        """Test finding audio files in directory."""
        directory = sample_audio_file.parent
        files = find_audio_files(directory)
        
        assert sample_audio_file in files
    
    def test_find_audio_files_recursive(self, tmp_path):
        """Test finding audio files recursively."""
        # Create nested directory structure with audio files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        # Create test audio files
        sr = 16000
        y = np.random.randn(sr)
        
        audio1 = tmp_path / "test1.wav"
        audio2 = subdir / "test2.wav"
        
        sf.write(str(audio1), y, sr)
        sf.write(str(audio2), y, sr)
        
        # Non-recursive should find only top-level file
        files = find_audio_files(tmp_path, recursive=False)
        assert len(files) == 1
        assert audio1 in files
        
        # Recursive should find both files
        files = find_audio_files(tmp_path, recursive=True)
        assert len(files) == 2
        assert audio1 in files
        assert audio2 in files
    
    def test_process_single_file_success(self, sample_audio_file, tmp_path):
        """Test processing single file successfully."""
        processor = AudioNet()
        logger = Mock()
        output_path = tmp_path / "output.wav"
        
        success = process_single_file(
            sample_audio_file,
            output_path,
            processor,
            logger,
            export_metrics=True
        )
        
        assert success is True
        assert output_path.exists()
        
        # Check metrics file was created
        metrics_path = output_path.with_suffix('.json')
        assert metrics_path.exists()
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert metrics["ok"] is True
    
    def test_process_single_file_failure(self, tmp_path):
        """Test processing single file with failure."""
        # Create a broken audio file
        broken_file = tmp_path / "broken.wav"
        broken_file.write_text("not audio data")
        
        processor = AudioNet()
        logger = Mock()
        
        success = process_single_file(
            broken_file,
            None,
            processor,
            logger,
            export_metrics=False
        )
        
        assert success is False
        logger.error.assert_called()
    
    @patch('sys.argv', ['audionet', '--help'])
    def test_main_help(self):
        """Test main function with help argument."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # argparse exits with code 0 for help
        assert exc_info.value.code == 0
    
    @patch('audionet.cli.find_audio_files')
    @patch('audionet.cli.process_single_file')
    @patch('sys.argv', ['audionet', 'input.wav'])
    def test_main_single_file(self, mock_process, mock_find_files, sample_audio_file):
        """Test main function with single file."""
        mock_find_files.return_value = [sample_audio_file]
        mock_process.return_value = True
        
        with patch('pathlib.Path.exists', return_value=True):
            main()
        
        mock_find_files.assert_called_once()
        mock_process.assert_called_once()
    
    @patch('audionet.cli.find_audio_files')
    @patch('sys.argv', ['audionet', 'nonexistent'])
    def test_main_nonexistent_input(self, mock_find_files):
        """Test main function with nonexistent input."""
        mock_find_files.return_value = []
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
    
    @patch('audionet.cli.find_audio_files')
    @patch('sys.argv', ['audionet', 'empty_dir'])
    def test_main_no_audio_files(self, mock_find_files):
        """Test main function with directory containing no audio files."""
        mock_find_files.return_value = []
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
    
    @patch('audionet.cli.find_audio_files')
    @patch('audionet.cli.process_single_file')
    @patch('sys.argv', ['audionet', 'input.wav', '--config', 'config.json'])
    def test_main_with_config(self, mock_process, mock_find_files, sample_audio_file, temp_config_file):
        """Test main function with custom config."""
        mock_find_files.return_value = [sample_audio_file]
        mock_process.return_value = True
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('audionet.cli.load_config') as mock_load_config:
                mock_load_config.return_value = AudioNetConfig(target_sr=8000)
                main()
        
        mock_load_config.assert_called_once()
    
    @patch('audionet.cli.find_audio_files')  
    @patch('audionet.cli.process_single_file')
    @patch('sys.argv', ['audionet', 'input.wav', '--progress'])
    def test_main_with_progress(self, mock_process, mock_find_files, sample_audio_file):
        """Test main function with progress enabled."""
        mock_find_files.return_value = [sample_audio_file]
        mock_process.return_value = True
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.print') as mock_print:
                main()
        
        # Should have progress output
        mock_print.assert_called()
    
    @patch('audionet.cli.find_audio_files')
    @patch('audionet.cli.process_single_file')
    @patch('sys.argv', ['audionet', 'input_dir', '--output', 'output_dir'])
    def test_main_batch_processing(self, mock_process, mock_find_files, tmp_path):
        """Test main function with batch processing."""
        # Create multiple test files
        audio_files = []
        for i in range(3):
            audio_file = tmp_path / f"test{i}.wav"
            audio_files.append(audio_file)
        
        mock_find_files.return_value = audio_files
        mock_process.return_value = True
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.mkdir'):
                main()
        
        # Should process all files
        assert mock_process.call_count == 3