"""
System utilities and hardware verification helpers for XAI-NIDS.
"""

from .gpu_info import get_system_environment_info, print_environment_report

__all__ = ["get_system_environment_info", "print_environment_report"]
