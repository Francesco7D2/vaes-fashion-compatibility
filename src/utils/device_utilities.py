"""Portable TensorFlow device detection and selection."""

import importlib.metadata
import os
import platform
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "vaes-fashion-matplotlib")
)

import tensorflow as tf


def choose_device(preference, gpu_count):
    """Resolve auto/cpu/gpu without depending on a particular OS."""
    preference = str(preference).lower()
    if preference not in {"auto", "cpu", "gpu"}:
        raise ValueError("training.device must be one of: auto, cpu, gpu")
    if preference == "gpu" and not gpu_count:
        raise RuntimeError(
            "GPU was requested but TensorFlow found none. Run "
            "`python -m scripts.check_device` and install the platform accelerator."
        )
    return (
        "gpu" if preference == "gpu" or (preference == "auto" and gpu_count) else "cpu"
    )


def configure_tensorflow(preference="auto"):
    """Configure devices before model creation and return a printable report."""
    gpus = tf.config.list_physical_devices("GPU")
    selected = choose_device(preference, len(gpus))
    notes = []
    tf.config.set_soft_device_placement(True)
    if selected == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            notes.append(f"GPU visibility was already initialized: {error}")
    else:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError) as error:
                # Metal manages memory itself; CUDA may support memory growth.
                notes.append(f"Memory growth unchanged for {gpu.name}: {error}")
    accelerator_packages = {}
    for package in ("tensorflow-metal", "nvidia-cudnn-cu12"):
        try:
            accelerator_packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and "tensorflow-metal" in accelerator_packages
        and not gpus
    ):
        notes.append(
            "tensorflow-metal is installed but no GPU is visible; auto will use CPU. "
            "Run outside a sandbox or use --device gpu to fail fast."
        )
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "preference": str(preference).lower(),
        "selected": selected,
        "device": "/GPU:0" if selected == "gpu" else "/CPU:0",
        "available_gpus": [gpu.name for gpu in gpus],
        "accelerator_packages": accelerator_packages,
        "notes": notes,
    }
