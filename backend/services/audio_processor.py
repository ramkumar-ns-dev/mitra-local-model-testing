import io
import os
import torch
import numpy as np
from fastapi import HTTPException, UploadFile
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from backend.config import settings
from backend.utils.logging_config import get_logger

logger = get_logger()

class AudioProcessor:
    @staticmethod
    def validate_file_metadata(filename: str, content_type: str = None, file_size: int = 0):
        """
        Validates file extension, MIME type, and file size before loading.
        """
        # Validate file size
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            logger.error(f"File size {file_size} bytes exceeds maximum limit of {settings.MAX_FILE_SIZE_BYTES} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB"
            )
        
        if file_size == 0:
            logger.error("Audio file is empty")
            raise HTTPException(status_code=400, detail="Empty audio file provided")

        # Validate file extension
        _, ext = os.path.splitext(filename.lower())
        if ext not in settings.ALLOWED_EXTENSIONS:
            logger.error(f"Unsupported file extension: {ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Validate MIME type if provided
        if content_type and content_type not in settings.ALLOWED_MIME_TYPES:
            # Note: Warn but do not strictly block if the extension is valid, as clients sometimes send generic/incorrect mime types
            logger.warning(f"Unexpected MIME type: {content_type} for file {filename}")

    @classmethod
    def process_audio(cls, file_bytes: bytes, filename: str) -> tuple[torch.Tensor, float]:
        """
        Processes audio bytes:
        1. Decodes WAV, MP3, M4A, FLAC using pydub.
        2. Validates duration.
        3. Converts to mono and resamples to target sample rate (16kHz).
        4. Converts to normalized float32 PyTorch tensor.
        
        Returns:
            tuple: (waveform_tensor, duration_seconds)
        """
        try:
            # Load audio using pydub (handles multiple formats)
            audio_io = io.BytesIO(file_bytes)
            
            # Extract format for pydub
            _, ext = os.path.splitext(filename.lower())
            audio_format = ext.replace(".", "")
            
            try:
                audio = AudioSegment.from_file(audio_io, format=audio_format)
            except (CouldntDecodeError, Exception) as decode_err:
                logger.warning(f"pydub decoding failed for {filename} with explicit format {audio_format}: {decode_err}. Retrying without format parameter...")
                # Retry letting pydub auto-detect
                audio_io.seek(0)
                audio = AudioSegment.from_file(audio_io)

        except Exception as e:
            logger.error(f"Failed to decode or parse audio file {filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail="Could not decode audio file. The file may be corrupted or in an unsupported format."
            )

        # Check duration
        duration_seconds = len(audio) / 1000.0
        logger.info(f"Loaded audio: {filename}, original rate: {audio.frame_rate}Hz, channels: {audio.channels}, duration: {duration_seconds:.2f}s")

        if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
            logger.error(f"Audio duration {duration_seconds:.1f}s exceeds maximum limit of {settings.MAX_AUDIO_DURATION_SECONDS}s")
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration exceeds maximum limit of {settings.MAX_AUDIO_DURATION_SECONDS} seconds"
            )

        if duration_seconds == 0:
            logger.error("Audio duration is 0 seconds")
            raise HTTPException(status_code=400, detail="Audio file contains no audio samples")

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to target rate (e.g., 16000Hz)
        if audio.frame_rate != settings.TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(settings.TARGET_SAMPLE_RATE)

        # Convert to raw float32 array
        samples = np.array(audio.get_array_of_samples())
        
        # Normalize based on sample width (bits per sample)
        if audio.sample_width == 2:
            # 16-bit PCM
            normalized_samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            # 32-bit PCM
            normalized_samples = samples.astype(np.float32) / 2147483648.0
        elif audio.sample_width == 1:
            # 8-bit PCM
            normalized_samples = (samples.astype(np.float32) - 128.0) / 128.0
        else:
            # Generic scaling
            max_val = float(1 << (8 * audio.sample_width - 1))
            normalized_samples = samples.astype(np.float32) / max_val

        # Convert to torch tensor
        waveform = torch.from_numpy(normalized_samples)
        
        return waveform, duration_seconds
