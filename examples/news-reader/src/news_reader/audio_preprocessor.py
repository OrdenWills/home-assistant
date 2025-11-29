from typing import Optional

import numpy as np
from scipy import signal


class AudioPreprocessor:
    """
    Preprocesses audio data with various filtering and normalization techniques.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio preprocessor.

        Args:
            sample_rate: Sample rate of the audio data
        """
        self.sample_rate = sample_rate

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to range [-1, 1].

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            # Already float, normalize by max absolute value
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

        return audio

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset (mean) from audio signal.

        Args:
            audio: Input audio array

        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)

    def apply_bandpass_filter(
        self,
        audio: np.ndarray,
        low_freq: float = 80,
        high_freq: float = 8000,
        order: int = 5,
    ) -> np.ndarray:
        """
        Apply bandpass filter to remove unwanted frequencies.

        Args:
            audio: Input audio array
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        sos = signal.butter(order, [low, high], btype="band", output="sos")
        filtered = signal.sosfilt(sos, audio)

        return filtered

    def reduce_noise(
        self, audio: np.ndarray, noise_sample: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction.

        Args:
            audio: Input audio array
            noise_sample: Optional noise profile (if None, uses first 0.5 seconds)

        Returns:
            Noise-reduced audio
        """
        if noise_sample is None:
            # Use first 0.5 seconds as noise profile
            noise_duration = int(0.5 * self.sample_rate)
            noise_sample = (
                audio[:noise_duration] if len(audio) > noise_duration else audio
            )

        # Estimate noise spectrum
        noise_fft = np.fft.rfft(noise_sample)
        noise_power = np.abs(noise_fft) ** 2

        # Process audio in chunks
        chunk_size = len(noise_sample)
        output = np.zeros_like(audio)

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Spectral subtraction
            chunk_fft = np.fft.rfft(chunk)
            chunk_power = np.abs(chunk_fft) ** 2

            # Subtract noise power spectrum
            clean_power = np.maximum(chunk_power - noise_power[: len(chunk_power)], 0)
            clean_magnitude = np.sqrt(clean_power)

            # Reconstruct signal
            clean_fft = clean_magnitude * np.exp(1j * np.angle(chunk_fft))
            clean_chunk = np.fft.irfft(clean_fft)

            output[i : i + len(clean_chunk[: len(audio) - i])] = clean_chunk[
                : len(audio) - i
            ]

        return output

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio: Input audio array
            threshold: Energy threshold for silence detection
            frame_length: Length of analysis frame
            hop_length: Hop size between frames

        Returns:
            Trimmed audio
        """
        # Calculate energy in frames
        energy = np.array(
            [
                np.sum(audio[i : i + frame_length] ** 2)
                for i in range(0, len(audio) - frame_length, hop_length)
            ]
        )

        # Find non-silent regions
        non_silent = energy > threshold

        if not np.any(non_silent):
            return audio  # Return original if all silent

        # Find start and end indices
        start_idx = np.argmax(non_silent) * hop_length
        end_idx = (
            len(non_silent) - np.argmax(non_silent[::-1]) - 1
        ) * hop_length + frame_length

        return audio[start_idx:end_idx]

    def preprocess(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        remove_dc: bool = True,
        bandpass: bool = True,
        denoise: bool = False,
        trim: bool = False,
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            audio: Input audio array
            normalize: Whether to normalize audio
            remove_dc: Whether to remove DC offset
            bandpass: Whether to apply bandpass filter
            denoise: Whether to apply noise reduction
            trim: Whether to trim silence

        Returns:
            Preprocessed audio
        """
        processed = audio.copy()

        if normalize:
            processed = self.normalize(processed)

        if remove_dc:
            processed = self.remove_dc_offset(processed)

        if bandpass:
            processed = self.apply_bandpass_filter(processed)

        if denoise:
            processed = self.reduce_noise(processed)

        if trim:
            processed = self.trim_silence(processed)

        return processed
