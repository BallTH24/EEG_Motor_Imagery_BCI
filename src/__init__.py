"""
Phase 3 & Phase 4: Continuous Asynchronous HMI Streaming Engine & Cyber-Physical Deployment Package
Brain Code Camp (BCC) 2026 - Track: Neurotechnology & Computational Neuroscience
"""

from .hmi_stream import (
    CircularRingBuffer,
    ContinuousHMIEngine,
    StreamSimulator,
    compute_crc8,
    encode_uart_frame,
    ESP32WatchdogEmulator,
    CMD_MAP,
    REVERSE_CMD_MAP,
)

__all__ = [
    "CircularRingBuffer",
    "ContinuousHMIEngine",
    "StreamSimulator",
    "compute_crc8",
    "encode_uart_frame",
    "ESP32WatchdogEmulator",
    "CMD_MAP",
    "REVERSE_CMD_MAP",
]
