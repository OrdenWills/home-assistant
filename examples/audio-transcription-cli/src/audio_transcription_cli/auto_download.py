"""Automatic download functionality for llama.cpp builds."""

import os
import stat
import subprocess
import shutil
import sys
from pathlib import Path
from .platform_utils import get_platform_info

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def validate_platform_support(platform_string: str) -> bool:
    """Check if the current platform is supported by available runners."""
    supported_platforms = [
        'android-arm64',
        'macos-arm64', 
        'ubuntu-arm64',
        'ubuntu-x64'
    ]
    
    return platform_string in supported_platforms


def fix_binary_permissions(target_dir: str) -> bool:
    """
    Fix execute permissions for binary files in runners directories.
    
    Args:
        target_dir: Base directory containing the downloaded files
        
    Returns:
        bool: True if permissions were fixed successfully
    """
    try:
        runners_dir = Path(target_dir) / "runners"
        if not runners_dir.exists():
            print("⚠️  No runners directory found, skipping permission fix")
            return True
        
        print("🔧 Fixing binary permissions...")
        
        # Find all binary files in runners subdirectories
        for platform_dir in runners_dir.iterdir():
            if platform_dir.is_dir():
                bin_dir = platform_dir / "bin"
                if bin_dir.exists():
                    for binary_file in bin_dir.iterdir():
                        if binary_file.is_file():
                            # Add execute permissions for owner, group, and others
                            current_permissions = binary_file.stat().st_mode
                            new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                            binary_file.chmod(new_permissions)
                            print(f"✅ Fixed permissions for: {binary_file}")
        
        print("🎯 Binary permissions fixed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Could not fix binary permissions: {e}")
        print("💡 You may need to manually run: chmod +x ./LFM2-Audio-1.5B-GGUF/runners/*/bin/*")
        return False


def clone_huggingface_repo(repo_url: str, target_dir: str) -> bool:
    """Download the Hugging Face repository with all files including LFS."""
    try:
        # Extract repo_id from URL
        repo_id = repo_url.replace("https://huggingface.co/", "")
        
        print(f"🔄 Downloading llama.cpp builds from: {repo_url}")
        
        # Try using huggingface_hub first (preferred method)
        if HF_HUB_AVAILABLE:
            print("📦 Using huggingface_hub for optimal download...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False  # Download actual files, not symlinks
            )
            print(f"✅ Successfully downloaded builds to {target_dir}")
            return True
        else:
            # Fallback to git clone with LFS
            print("🔄 Using git clone with LFS (installing huggingface_hub is recommended)...")
            
            # First clone the repository
            subprocess.run([
                'git', 'clone', repo_url, target_dir
            ], check=True, capture_output=True)
            
            # Then pull LFS files
            subprocess.run([
                'git', 'lfs', 'pull'
            ], cwd=target_dir, check=True, capture_output=True)
            
            print(f"✅ Successfully downloaded builds to {target_dir}")
            return True
            
    except Exception as e:
        print(f"❌ Error downloading repository: {e}")
        print("💡 Try installing huggingface_hub: pip install huggingface_hub")
        return False


def download_llama_cpp_builds_for_audio() -> bool:
    """
    Automatically download llama.cpp builds for LFM2-Audio-1.5B model.
    
    Returns:
        bool: True if builds are available (either already existed or successfully downloaded),
              False if download failed.
    """
    
    # Configuration
    REPO_URL = "https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF"
    TARGET_DIR = "LFM2-Audio-1.5B-GGUF"
    
    # Detect current platform first
    current_platform = get_platform_info()
    print(f"🔍 Detected platform: {current_platform}")
    
    # Validate platform support (blocking - exit if not supported)
    if not validate_platform_support(current_platform):
        print(f"❌ ERROR: Your platform ({current_platform}) is not supported.")
        print("   Supported platforms: android-arm64, macos-arm64, ubuntu-arm64, ubuntu-x64")
        print("   Please wait for builds to be released for your platform.")
        sys.exit(1)
    
    print(f"✅ Platform {current_platform} is supported!")
    
    # Check if target directory already exists
    if os.path.exists(TARGET_DIR):
        runners_dir = Path(TARGET_DIR) / "runners"
        if runners_dir.exists():
            print(f"✅ llama.cpp builds already available at: {runners_dir}")

            # Fix binary permissions just in case
            fix_binary_permissions(TARGET_DIR)

            return True
        else:
            print(f"🧹 Removing incomplete download directory: {TARGET_DIR}")
            shutil.rmtree(TARGET_DIR)
    
    # Clone the repository
    success = clone_huggingface_repo(REPO_URL, TARGET_DIR)

    if success:
                
        runners_dir = Path(TARGET_DIR) / "runners"
        if runners_dir.exists():
            print(f"🎯 llama.cpp builds ready! Runners available at: {runners_dir}")

            # Fix binary permissions after download
            fix_binary_permissions(TARGET_DIR)

            return True
        else:
            print(f"⚠️  Warning: runners directory not found in {TARGET_DIR}")
            print("    The repository structure may have changed.")
            return False
    else:
        print("❌ Failed to download llama.cpp builds.")
        print("💡 You can try running 'uv run download_llama_cpp_builds.py' manually.")
        return False