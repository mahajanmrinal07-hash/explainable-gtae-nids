"""
Unit tests for hardware and GPU reporting utility.
"""

from src.utils.gpu_info import get_system_environment_info


def test_gpu_info_structure():
    info = get_system_environment_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_count_physical" in info
    assert "ram_total_gb" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info

    if info["cuda_available"]:
        assert info["gpu_count"] >= 1
        assert len(info["gpus"]) >= 1
        gpu = info["gpus"][0]
        assert "name" in gpu
        assert "total_vram_gb" in gpu
        assert "free_vram_gb" in gpu
        assert gpu["total_vram_gb"] > 0
