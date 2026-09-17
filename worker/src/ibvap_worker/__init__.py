"""IBVAP analytics worker.

SIH 2026 · PS 26187 · Team SW-73 (ByteForge).

The pipeline, in order (BUILD_SPEC §3.3):

    ingest -> evqm -> enhance -> detect -> track
           -> geometry -> rules -> risk -> debounce -> evidence -> sinks

Read docs/BUILD_SPEC.md §7 before changing any public signature in this package.
Those contracts are frozen.
"""

__version__ = "1.0.0"
__spec_version__ = "1.0.0"
