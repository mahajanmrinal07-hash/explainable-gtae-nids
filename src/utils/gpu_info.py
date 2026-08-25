"""
Hardware and Environment Inspection Utility.
Reports CPU, GPU, VRAM, CUDA, and PyTorch status.
"""

import os
import platform
import psutil
import torch
from typing import Dict, Any


def get_system_environment_info() -> Dict[str, Any]:
    """
    Collect comprehensive information about the host CPU, GPU, CUDA, and PyTorch runtime.
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpu_data = {
                "index": i,
                "name": props.name,
                "total_vram_gb": round(total_mem / (1024 ** 3), 2),
                "total_vram_mb": round(total_mem / (1024 ** 2), 2),
                "free_vram_gb": round(free_mem / (1024 ** 3), 2),
                "free_vram_mb": round(free_mem / (1024 ** 2), 2),
                "used_vram_mb": round((total_mem - free_mem) / (1024 ** 2), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
            info["gpus"].append(gpu_data)

    return info


def print_environment_report() -> None:
    """
    Prints a formatted report of system hardware and PyTorch CUDA environment.
    """
    env = get_system_environment_info()

    print("=" * 65)
    print("      XAI-NIDS HARDWARE & RUNTIME ENVIRONMENT REPORT")
    print("=" * 65)
    print(f" OS                  : {env['os']} {env['os_release']} ({env['os_version']})")
    print(f" Python Version      : {env['python_version']}")
    print(f" Physical CPU Cores  : {env['cpu_count_physical']}")
    print(f" Logical CPU Cores   : {env['cpu_count_logical']}")
    print(f" Total System RAM    : {env['ram_total_gb']} GB (Available: {env['ram_available_gb']} GB)")
    print("-" * 65)
    print(f" PyTorch Version     : {env['pytorch_version']}")
    print(f" CUDA Available      : {env['cuda_available']}")
    print(f" PyTorch CUDA Build  : {env['cuda_version'] if env['cuda_version'] else 'N/A'}")
    print(f" GPU Device Count    : {env['gpu_count']}")

    if env["cuda_available"] and env["gpus"]:
        for gpu in env["gpus"]:
            print("-" * 65)
            print(f" GPU [{gpu['index']}] Name         : {gpu['name']}")
            print(f"   Total VRAM        : {gpu['total_vram_gb']} GB ({gpu['total_vram_mb']} MiB)")
            print(f"   Free VRAM         : {gpu['free_vram_gb']} GB ({gpu['free_vram_mb']} MiB)")
            print(f"   Used VRAM         : {gpu['used_vram_mb']} MiB")
            print(f"   Compute Capability: {gpu['compute_capability']}")
            print(f"   SM Count          : {gpu['multi_processor_count']}")
    else:
        print(" [!] No NVIDIA CUDA GPU detected or PyTorch CUDA runtime is inactive.")
    print("=" * 65)


if __name__ == "__main__":
    print_environment_report()
