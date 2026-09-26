import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime


OUTPUT_DIR = Path("environment_snapshot")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )

        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {
            "command": command,
            "error": str(e),
        }


def get_module_version(module_name):
    try:
        module = __import__(module_name)

        return {
            "installed": True,
            "version": getattr(module, "__version__", "unknown"),
            "file": getattr(module, "__file__", "unknown"),
        }
    except Exception as e:
        return {
            "installed": False,
            "error": str(e),
        }


print("=" * 70)
print("ExtractHQ GPU Environment Snapshot")
print("=" * 70)

# ------------------------------------------------------------------
# Basic system information
# ------------------------------------------------------------------

system_info = {
    "timestamp": datetime.now().isoformat(),
    "python": {
        "version": sys.version,
        "executable": sys.executable,
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    },
    "environment": {
        key: value
        for key, value in os.environ.items()
        if key in [
            "PATH",
            "PYTHONPATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VERSION",
            "LD_LIBRARY_PATH",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
        ]
    },
}


with open(OUTPUT_DIR / "system.json", "w") as f:
    json.dump(system_info, f, indent=2)


# ------------------------------------------------------------------
# Important Python packages
# ------------------------------------------------------------------

print("\n[1] Checking important packages...")

packages = [
    "paddle",
    "paddleocr",
    "paddlex",
    "torch",
    "torchvision",
    "torchaudio",
    "vllm",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "opencv",
    "cv2",
    "PIL",
    "requests",
]

package_info = {}

for package in packages:
    info = get_module_version(package)
    package_info[package] = info

    if info["installed"]:
        print(f"  {package}: {info['version']}")
    else:
        print(f"  {package}: NOT IMPORTABLE")


with open(OUTPUT_DIR / "important_packages.json", "w") as f:
    json.dump(package_info, f, indent=2)


# ------------------------------------------------------------------
# pip freeze
# ------------------------------------------------------------------

print("\n[2] Capturing pip freeze...")

pip_freeze = run_command(
    f'"{sys.executable}" -m pip freeze'
)

with open(OUTPUT_DIR / "requirements.txt", "w") as f:
    f.write(pip_freeze.get("stdout", ""))

print("  Saved: requirements.txt")


# ------------------------------------------------------------------
# pip check
# ------------------------------------------------------------------

print("\n[3] Running pip check...")

pip_check = run_command(
    f'"{sys.executable}" -m pip check'
)

with open(OUTPUT_DIR / "pip-check.txt", "w") as f:
    f.write("STDOUT\n")
    f.write(pip_check.get("stdout", ""))
    f.write("\nSTDERR\n")
    f.write(pip_check.get("stderr", ""))

print("  Saved: pip-check.txt")


# ------------------------------------------------------------------
# pip show for critical packages
# ------------------------------------------------------------------

print("\n[4] Capturing package metadata...")

critical_packages = [
    "paddlepaddle-gpu",
    "paddleocr",
    "paddlex",
    "torch",
    "torchvision",
    "torchaudio",
    "vllm",
    "transformers",
    "huggingface-hub",
    "fastapi",
    "uvicorn",
]

package_show = {}

for package in critical_packages:
    result = run_command(
        f'"{sys.executable}" -m pip show "{package}"'
    )

    package_show[package] = result


with open(OUTPUT_DIR / "package-metadata.json", "w") as f:
    json.dump(package_show, f, indent=2)


# ------------------------------------------------------------------
# NVIDIA / CUDA
# ------------------------------------------------------------------

print("\n[5] Capturing NVIDIA/CUDA information...")

commands = {
    "nvidia-smi": "nvidia-smi",
    "nvidia-smi-query": (
        "nvidia-smi --query-gpu="
        "name,driver_version,memory.total,"
        "compute_cap,temperature.gpu,power.draw "
        "--format=csv"
    ),
    "nvcc": "nvcc --version",
    "ldconfig-cuda": "ldconfig -p | grep -E 'cuda|cudnn|cublas|nccl' || true",
}

gpu_info = {}

for name, command in commands.items():
    print(f"  Running: {command}")
    gpu_info[name] = run_command(command)


with open(OUTPUT_DIR / "gpu-cuda.json", "w") as f:
    json.dump(gpu_info, f, indent=2)


# ------------------------------------------------------------------
# Paddle runtime test
# ------------------------------------------------------------------

print("\n[6] Testing Paddle GPU...")

paddle_test = run_command(
    f'''"{sys.executable}" -c "
import paddle
print('Paddle:', paddle.__version__)
print('CUDA:', paddle.is_compiled_with_cuda())
print('Device:', paddle.device.get_device())

x = paddle.randn([1024, 1024])
y = paddle.randn([1024, 1024])
z = paddle.matmul(x, y)

print('Matmul:', z.shape)
print('GPU test: OK')
"'''
)

with open(OUTPUT_DIR / "paddle-test.txt", "w") as f:
    f.write(paddle_test.get("stdout", ""))
    f.write("\n")
    f.write(paddle_test.get("stderr", ""))


# ------------------------------------------------------------------
# PyTorch runtime test
# ------------------------------------------------------------------

print("\n[7] Testing PyTorch GPU...")

torch_test = run_command(
    f'''"{sys.executable}" -c "
import torch

print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    z = torch.matmul(x, y)

    print('Matmul:', z.shape)
    print('GPU test: OK')
else:
    print('GPU test: FAILED')
"'''
)

with open(OUTPUT_DIR / "torch-test.txt", "w") as f:
    f.write(torch_test.get("stdout", ""))
    f.write("\n")
    f.write(torch_test.get("stderr", ""))


# ------------------------------------------------------------------
# vLLM version
# ------------------------------------------------------------------

print("\n[8] Testing vLLM...")

vllm_test = run_command(
    f'''"{sys.executable}" -c "
import vllm
print('vLLM:', vllm.__version__)
"'''
)

with open(OUTPUT_DIR / "vllm-test.txt", "w") as f:
    f.write(vllm_test.get("stdout", ""))
    f.write("\n")
    f.write(vllm_test.get("stderr", ""))


# ------------------------------------------------------------------
# Environment variables relevant to ML
# ------------------------------------------------------------------

print("\n[9] Capturing ML environment variables...")

ml_env = {}

for key, value in os.environ.items():
    key_upper = key.upper()

    if any(
        term in key_upper
        for term in [
            "CUDA",
            "NVIDIA",
            "PADDLE",
            "PADDLEX",
            "VLLM",
            "TORCH",
            "PYTHON",
            "HF_",
            "HUGGINGFACE",
            "TRANSFORMERS",
            "MODEL",
            "LD_LIBRARY",
        ]
    ):
        ml_env[key] = value


with open(OUTPUT_DIR / "ml-environment.json", "w") as f:
    json.dump(ml_env, f, indent=2)


# ------------------------------------------------------------------
# Installed Python executable
# ------------------------------------------------------------------

with open(OUTPUT_DIR / "python-path.txt", "w") as f:
    f.write(sys.executable + "\n")


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

summary = {
    "snapshot_created": datetime.now().isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "important_packages": package_info,
    "files": [
        "system.json",
        "important_packages.json",
        "requirements.txt",
        "pip-check.txt",
        "package-metadata.json",
        "gpu-cuda.json",
        "paddle-test.txt",
        "torch-test.txt",
        "vllm-test.txt",
        "ml-environment.json",
        "python-path.txt",
    ],
}


with open(OUTPUT_DIR / "SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)


print("\n" + "=" * 70)
print("SNAPSHOT COMPLETE")
print("=" * 70)
print(f"Directory: {OUTPUT_DIR.absolute()}")
print()
print("Files:")
for file in summary["files"]:
    print(f"  {OUTPUT_DIR / file}")

print("\nUse requirements.txt as the primary dependency reference.")
print("=" * 70)