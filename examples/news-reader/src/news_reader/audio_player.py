
from pathlib import Path
import time


class AudioPlayer:
    """Audio player for news reader application using pygame."""
    
    def __init__(self):
        """Initialize the audio player."""
        self.mixer = None
        self._initialize_pygame()
    
    def _initialize_pygame(self) -> None:
        """Initialize pygame mixer for audio playback."""
        try:
            import pygame
            pygame.mixer.init()
            self.mixer = pygame.mixer
            print("Audio player initialized successfully")
        except ImportError:
            print("Warning: pygame not installed. Audio playback will be disabled.")
            self.mixer = None
        except Exception as e:
            print(f"Warning: Could not initialize audio player: {e}")
            self.mixer = None
    
    def play(self, audio_path: str, wait_for_completion: bool = True) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_path: Path to the audio file to play
            wait_for_completion: If True, wait for audio to finish playing
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if self.mixer is None:
            print("Audio player not available")
            return False
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"Audio file not found: {audio_path}")
            return False
        
        try:
            print(f"Playing audio: {audio_path}")
            self.mixer.music.load(str(audio_file))
            self.mixer.music.play()
            
            if wait_for_completion:
                # Wait for playback to complete
                while self.mixer.music.get_busy():
                    time.sleep(0.1)
                print("Audio playback completed")
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def stop(self) -> None:
        """Stop audio playback."""
        if self.mixer is not None:
            self.mixer.music.stop()
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if self.mixer is None:
            return False
        return self.mixer.music.get_busy()
    
    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if self.mixer is not None:
            self.mixer.quit()
