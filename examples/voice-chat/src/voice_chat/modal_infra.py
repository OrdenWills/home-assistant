import modal


def get_modal_app(name: str) -> modal.App:
    """
    Returns the Modal application object.
    """
    return modal.App(name)


def get_docker_image() -> modal.Image:
    """
    Returns a Modal Docker image with all the required Python dependencies installed.
    """
    docker_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "ffmpeg",
            "libavcodec59",      # Changed from 58
            "libavformat59",     # Changed from 58
            "libavutil57",       # Changed from 56
            "libswscale6",       # Changed from 5
            "portaudio19-dev",   # PortAudio development headers for PyAudio
            "libasound2-dev",    # ALSA development headers
            "python3-dev"       # Python development headers
        )
        .uv_pip_install(
            "liquid-audio>=1.0.0",
            "modal>=1.2.4",
            "torchcodec>=0.8.1",
            "numpy>=2.3.5",
            "pyaudio>=0.2.14",
            "rich>=13.0.0",
            "scipy>=1.16.3",
        )
        # .add_local_python_source(".")
        .env({"HF_HOME": "/model_cache"})
    )

    return docker_image


def get_volume(name: str) -> modal.Volume:
    """
    Returns a Modal volume object for the given name.
    """
    return modal.Volume.from_name(name, create_if_missing=True)


def get_retries(max_retries: int) -> modal.Retries:
    """
    Returns the retry policy for failed tasks.
    """
    return modal.Retries(initial_delay=0.0, max_retries=max_retries)


def get_secrets() -> list[modal.Secret]:
    """
    Returns the Weights & Biases secret.
    """
    wandb_secret = modal.Secret.from_name("wandb-secret")
    return [wandb_secret]

