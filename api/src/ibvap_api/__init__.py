"""DRISHTI-BOP API.

REST + WebSocket surface over the alerts the worker produces (BUILD_SPEC §8, §9).
Async throughout — this layer is I/O bound, unlike the worker's threaded hot
path. Do not mix the two models (CLAUDE.md/Code).
"""

__version__ = "1.0.0"
