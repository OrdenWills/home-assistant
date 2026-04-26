#!/usr/bin/env python
"""
Download models to local cache before running the app.
This avoids timeouts when selecting models in the UI.

Prioritizes smaller models first. If interrupted, simply rerun to resume downloads.
The script skips already-cached files automatically.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Models to download - ORDERED BY SIZE (smallest first for faster initial setup)
MODELS = [
    {
        "id": "home-assistant-sft-small-8bit",
        "name": "Home Assistant SFT (Finetuned 350M)",
        "repo": "OrdenWills/LFM2.5-350M-home-assistant-sft",
        "file": "LFM2.5-350M-home-assistant-sft-stage2.Q8_0.gguf",             # Changed from Q4_K_M to Q8_0
        "priority": 0,
    },
    {
    "id": "lfm2-vl-450m-q4",
    "name": "LFM2-VL-450M Q4",
    "repo": "LiquidAI/LFM2-VL-450M-GGUF",
    "file": "LFM2-VL-450M-Q4_0.gguf",
    "priority": 4,
},
    {
        "id": "lfm2.5-350m-q4",
        "name": "LFM2.5-350M Q4",
        "repo": "LiquidAI/LFM2.5-350M-GGUF",
        "file": "LFM2.5-350M-Q4_0.gguf",
        "priority":6,
    },
    {
        "id": "lfm2.5-1.2b-thinking-q4",
        "name": "LFM2.5-1.2B-Thinking-Q4",
        "repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "file": "LFM2.5-1.2B-Thinking-Q4_0.gguf",
        "priority": 3,
    },
    {
        "id": "lfm2.5-1.2b-thinking-q8",
        "name": "LFM2.5-1.2B-Thinking-Q8",
        "repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "file": "LFM2.5-1.2B-Thinking-Q8_0.gguf",
        "priority": 2,
    },   
    {
        "id": "home-assistant-sft",
        "name": "Home Assistant SFT (Finetuned)",
        "repo": "OrdenWills/LFM2.5-1.2B-home-assistant-sft",
        "file": "LFM2.5-1.2B-Instruct.Q4_K_M.gguf",
        "priority": 5,
    },
    {
        "id": "home-assistant-sft(small)",
        "name": "Home Assistant SFT (Finetuned)",
        "repo": "OrdenWills/LFM2.5-350M-home-assistant-sft",
        "file": "LFM2.5-350M.Q4_K_M.gguf",
        "priority": 1,
    },
]


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def is_file_complete(path):
    """Check if downloaded file appears to be complete (has reasonable size)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    # All our models are > 300MB, a file under 100MB is likely incomplete
    return size > 100 * 1024 * 1024


def download_models():
    """Download all models to HuggingFace cache, resuming interrupted downloads."""
    print("[ModelDownloader] Starting downloads...")
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"[ModelDownloader] Cache location: {cache_dir}")
    print(f"[ModelDownloader] Models ordered by size (smallest first)")
    print()
    
    # Sort by priority
    models_sorted = sorted(MODELS, key=lambda m: m["priority"])
    
    completed = 0
    skipped = 0
    failed = 0
    
    for model in models_sorted:
        print(f"[{completed + skipped + failed + 1}/{len(MODELS)}] Downloading {model['name']}...")
        print(f"  Repo: {model['repo']}")
        print(f"  File: {model['file']}")
        
        try:
            path = hf_hub_download(
                repo_id=model["repo"],
                filename=model["file"],
                repo_type="model",
                local_files_only=False,
            )
            
            if is_file_complete(path):
                file_size = Path(path).stat().st_size
                print(f"  ✓ Complete: {path}")
                print(f"  ✓ Size: {format_size(file_size)}")
                completed += 1
            else:
                print(f"  ⚠ File appears incomplete, will retry on next run")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  ⚠ Resuming this download on next run will retry automatically")
            failed += 1
        
        print()
    
    print("=" * 50)
    print(f"[ModelDownloader] Downloads finished!")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    print("=" * 50)
    
    if failed == 0:
        print("✓ All models cached! You can now start the app without UI timeouts.")
    else:
        print("! Some downloads failed. Rerun this script to resume:")
        print("  python download_models.py")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = download_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[ModelDownloader] Download interrupted by user")
        print("[ModelDownloader] Rerun to resume: python download_models.py")
        sys.exit(1)
