"""SIL_HACKATHON loop engine: adapters that connect the pipeline stage YAMLs (00..10) into one run.

Entry point: ``python run.py --mock`` (no CARLA, no network) or ``python run.py --real``.
Stage YAMLs stay the source of truth; this package only reads them.
"""
