from .modal_infra import get_volume


class AudioFileManager:
    """
    API for managing file uploads and downloads to/from Modal volumes.
    """
    
    def __init__(self, volume_name: str, session_id: str):
        """
        Initialize the AudioFileManager with a volume name and session ID.
        
        Args:
            volume_name: Name of the Modal volume
            session_id: Unique session identifier for organizing files
        """
        self.volume_name = volume_name
        self.session_id = session_id
        self.volume = get_volume(volume_name)
        self.session_dir = f"/{session_id}"
    
    def upload(self, local_file_path: str, filename: str = "question.wav") -> None:
        """
        Upload a local WAV file to the Modal volume in the session directory.
        
        Args:
            local_file_path: Path to the local WAV file
            filename: Name for the file in the volume (default: "question.wav")
        """
        from pathlib import Path
        
        local_path = Path(local_file_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        if not local_path.suffix.lower() == '.wav':
            raise ValueError(f"File must be a WAV file, got: {local_path.suffix}")
        
        # Ensure session directory exists and construct remote path
        self.create_session_directory()
        remote_path = f"{self.session_dir}/{filename}"
        
        with self.volume.batch_upload() as batch:
            batch.put_file(local_file_path, remote_path)
        
        print(f"Uploaded {local_file_path} to volume '{self.volume_name}' at {remote_path}")
    
    def download(self, filename: str, local_file_path: str, max_retries: int = 5, retry_delay: float = 1.0) -> None:
        """
        Download a WAV file from the Modal volume session directory to local filesystem.
        
        Args:
            filename: Name of the file in the session directory (e.g., "answer1.wav")
            local_file_path: Where to save the file locally
            max_retries: Maximum number of retry attempts (default: 5)
            retry_delay: Delay between retries in seconds (default: 2.0)
        """
        from pathlib import Path
        import time
        
        # Create local directory if it doesn't exist
        local_path = Path(local_file_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Construct remote path using session directory
        remote_path = f"{self.session_dir}/{filename}"
        
        # For now, we'll use modal CLI for downloading since the Python SDK download
        # methods aren't clearly documented in the fetched docs
        import subprocess
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"Downloading {filename} (attempt {attempt + 1}/{max_retries})...")
                
                subprocess.run([
                    "modal", "volume", "get", 
                    self.volume_name, 
                    remote_path, 
                    local_file_path
                ], capture_output=True, text=True, check=True)
                
                print(f"Downloaded {remote_path} from volume '{self.volume_name}' to {local_file_path}")
                return  # Success - exit the retry loop
                
            except subprocess.CalledProcessError as e:
                last_error = e
                error_msg = e.stderr.strip() if e.stderr else "Unknown error"
                
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    print(f"Download failed (attempt {attempt + 1}): {error_msg}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Download failed on final attempt: {error_msg}")
        
        # If we get here, all retries failed
        raise RuntimeError(f"Failed to download file after {max_retries} attempts. Last error: {last_error.stderr if last_error else 'Unknown'}")
    
    def create_session_directory(self) -> str:
        """
        Create a session directory in the volume for organizing files.
            
        Returns:
            The session directory path in the volume
        """
        # Create empty directory marker file to ensure directory exists
        marker_path = f"{self.session_dir}/.session_marker"
        
        with self.volume.batch_upload() as batch:
            import io
            batch.put_file(io.BytesIO(b""), marker_path)
        
        return self.session_dir